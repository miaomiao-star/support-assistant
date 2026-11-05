"""Ch09正式图：知识闸不移位；生成自评、当轮快照与反馈凭据独立保存。"""

from copy import deepcopy
import re
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from app.workflow import tooling, conversation, routing, graph as base
from app.learning.confidence import assess, snapshot
from app.learning.observability import observation
from app.learning.prompts import Sufficiency, SUFFICIENCY_PROMPT
from app.knowledge.mining import structured
from app.rag.quality import REFUSAL, Answer, validate_answer


class State(tooling.State, total=False):
    retrieval_ran: bool
    retrieval_ranked: list
    retrieved_chunks: list | None
    evidence_confidence: dict
    learning_turns: dict
    self_check: dict | None


async def resolve(state, runtime):
    updates = await tooling.resolve(state, runtime)
    return {
        **updates,
        'retrieval_ran': False,
        'retrieval_ranked': [],
        'retrieved_chunks': None,
        'evidence_confidence': {},
        'self_check': None,
    }


async def classify(state, runtime):
    updates = await tooling.classify(state, runtime)
    runtime.context.learning.tracing.intent(updates['intent'], updates['operation'])
    return updates


async def retrieve(state, runtime):
    registry = runtime.context.registry
    registry.capture_retrieval = lambda result: setattr(
        registry, 'latest_retrieval', deepcopy(result)
    )
    with observation(
        'knowledge-retrieval',
        {'queries': state.get('queries'), 'question': state['normalized_question']},
        kind='retriever',
    ) as span:
        updates = await conversation.retrieve(state, runtime)
        result = registry.latest_retrieval
        span.update(output=result)
    return {
        **updates,
        'retrieval_ran': True,
        'retrieval_ranked': result['ranked'],
        'retrieved_chunks': snapshot(result['ranked'], retrieved=True),
    }


def gate(state, runtime):
    confidence = assess(state['retrieval_ranked'], runtime.context.learning.profile)
    allowed = confidence['allowed'] and bool(state['evidence'])
    get_stream_writer()(
        dict(type='citations', citations=state['evidence'], cited=[], useful=allowed)
    )
    return dict(
        gate_allowed=allowed,
        evidence_confidence=confidence,
        trace=base.mark(state, 'gate', **confidence),
    )


async def refuse(state, runtime):
    await runtime.context.rag.quality.low_confidence(
        runtime.context.conversation_id,
        state['raw_question'],
        'retrieval_low_conf',
        'Ch09证据闸：' + str(state['evidence_confidence']),
        retrieved_chunks=state['retrieved_chunks'],
    )
    get_stream_writer()(dict(type='delta', delta=REFUSAL))
    return dict(
        answer=REFUSAL,
        actions=['handoff', 'create_ticket'],
        stop_reason='weak_evidence',
        trace=base.mark(state, 'refuse'),
    )


async def agent(state, runtime):
    # 仅知识支路缓冲可见文本；工具申请/观察继续按原图执行，不绕过统一引擎。
    registry = runtime.context.registry
    registry.defer_text = bool(state.get('retrieval_ran'))
    tooling.prepare_registry(state, registry)
    original_bundle = registry.extra_bundle

    def bundle(current):
        result = original_bundle(current)
        if current.get('route') == 'knowledge':
            result['mode'] = 'knowledge'
        return result

    # 只给知识支路明确模式；订单政策、业务工具和工单收集仍用Ch08的原模式。
    registry.extra_bundle = bundle
    return await conversation.agent(state, runtime)


async def answer_check(state, runtime):
    if not state.get('retrieval_ran') or state.get('stop_reason'):
        return {}
    context = runtime.context
    if getattr(context.learning, 'assessor', None):
        review = await context.learning.assessor(state)
        verdict = Sufficiency.model_validate(review)
    else:
        verdict = await structured(
            context.model,
            Sufficiency,
            SUFFICIENCY_PROMPT,
            dict(
                question=state['raw_question'],
                draft=state['answer'],
                evidence=state['evidence'],
                confirmed_order=state.get('order_snapshot'),
                business_observations=[
                    m.content
                    for m in state['messages']
                    if m.type == 'tool'
                    and (m.id or '').startswith('react:' + state['request_id'])
                ],
            ),
        )
    cited = [int(n) for n in re.findall(r'\[(\d+)\]', state['answer'])]
    checked = validate_answer(
        Answer(
            useful=verdict.useful,
            reason=verdict.reason,
            answer=state['answer'],
            citations=cited,
        ),
        state['evidence'],
    )
    answer = checked.answer if checked.useful else REFUSAL
    if not checked.useful:
        await context.rag.quality.low_confidence(
            context.conversation_id,
            state['raw_question'],
            'self_check',
            checked.reason,
            retrieved_chunks=state['retrieved_chunks'],
        )
    get_stream_writer()(dict(type='delta', delta=answer))
    return dict(
        answer=answer,
        self_check=checked.model_dump(),
        messages=[AIMessage(content=answer, id='final:' + state['request_id'])],
    )


async def log_result(state, runtime):
    updates = await tooling.log_result(state, runtime)
    final = updates['messages'][-1]
    records = dict(state.get('learning_turns', {}))
    records[state['request_id']] = dict(
        feedback_id=state['request_id'],
        answer_message_id=str(final.additional_kwargs['sql_msg_id']),
        raw_question=state['raw_question'],
        retrieved=state.get('retrieval_ran', False),
        retrieved_chunks=deepcopy(state.get('retrieved_chunks')),
        intent=state['intent'],
        operation=state.get('operation'),
    )
    return {**updates, 'learning_turns': records}


def build_graph(checkpointer):
    graph = StateGraph(State, context_schema=conversation.Context)
    nodes = dict(
        resolve=resolve,
        classify=classify,
        route=tooling.route,
        order_slot=routing.order_slot,
        pick_order=routing.pick_order,
        load_order=tooling.load_order,
        expand=routing.expand,
        retrieve=retrieve,
        gate=gate,
        refuse=refuse,
        agent=agent,
        tools=tooling.tools,
        ticket_preview=tooling.ticket_preview,
        ticket_execute=tooling.ticket_execute,
        answer_check=answer_check,
        log=log_result,
        complaint=base.complaint,
        smalltalk=base.smalltalk,
    )
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, 'resolve')
    graph.add_edge('resolve', 'classify')
    graph.add_edge('classify', 'route')
    graph.add_conditional_edges(
        'route',
        lambda s: s['route'],
        dict(
            order='order_slot',
            knowledge='retrieve',
            complaint='complaint',
            smalltalk='smalltalk',
            other='agent',
        ),
    )
    graph.add_conditional_edges(
        'order_slot',
        lambda s: bool(s['selected_order_id']),
        {True: 'load_order', False: 'pick_order'},
    )
    graph.add_conditional_edges(
        'pick_order',
        lambda s: (
            'cancelled'
            if s.get('stop_reason') == 'selection_cancelled'
            else tooling.after_order(s)
        ),
        dict(cancelled=END, core='load_order', business='load_order'),
    )
    graph.add_conditional_edges(
        'load_order', tooling.after_order, dict(core='expand', business='agent')
    )
    graph.add_edge('expand', 'retrieve')
    graph.add_edge('retrieve', 'gate')
    graph.add_conditional_edges(
        'gate', lambda s: s['gate_allowed'], {True: 'agent', False: 'refuse'}
    )
    graph.add_conditional_edges(
        'agent',
        lambda s: bool(s.get('pending_call')),
        {True: 'tools', False: 'answer_check'},
    )
    graph.add_conditional_edges(
        'tools',
        lambda s: bool(s.get('ticket_preview')),
        {True: 'ticket_preview', False: 'agent'},
    )
    graph.add_edge('ticket_preview', 'ticket_execute')
    graph.add_edge('ticket_execute', 'agent')
    graph.add_edge('answer_check', 'log')
    for name in ('refuse', 'complaint', 'smalltalk'):
        graph.add_edge(name, 'log')
    graph.add_edge('log', END)
    compiled = graph.compile(checkpointer=checkpointer)
    compiled.formal_router = compiled.layered_context = compiled.unified_tools = (
        compiled.learning_tools
    ) = True
    return compiled
