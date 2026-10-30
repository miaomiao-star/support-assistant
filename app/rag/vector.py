"""Standalone原生BM25+Dense集合，复用前章同步线程与取消收敛封装。"""

from pymilvus import DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from app.knowledge.vector import MilvusVectors
from app.knowledge.splitter import embedding_text
from app.rag.retrieval import build_filter


class HybridVectors(MilvusVectors):
    def __init__(self, uri, collection='knowledge_ch04', metadata=None):
        super().__init__(uri, collection)
        self.metadata = metadata or {}
        # 只缓存已核验并加载的维度；连接关闭后必须清掉，避免重连时跳过加载。
        self.ready_dim = None

    def product_category(self, row):
        """品类只从显式型号/章节标注取值，不写回正文或生成额外知识副本。"""
        path = row.get('section_path') or ''
        for token, category in self.metadata.items():
            if token in path or token in row['questions']:
                return category
        return '通用'

    def _ready(self, dim, create=False):
        # 本方法只在持有父类线程锁的同步函数里调用，不直接阻塞FastAPI事件循环。
        from pymilvus import MilvusClient

        if self.client is None:
            self.client = MilvusClient(uri=self.uri, timeout=30)
        if self.ready_dim == dim:
            return True
        if not self.client.has_collection(self.collection):
            # 查询不会偷偷创建空集合；只有建库upsert入口允许创建。
            if not create:
                return False
            schema = self.client.create_schema(
                auto_id=False, enable_dynamic_field=False
            )
            schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field('dense', DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field(
                'text',
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={'type': 'chinese'},
            )
            # sparse由Milvus的BM25 Function生成，应用不自行分词或计算稀疏向量。
            schema.add_field('sparse', DataType.SPARSE_FLOAT_VECTOR)
            for name in ('category', 'content_type', 'product_category'):
                schema.add_field(name, DataType.VARCHAR, max_length=255)
            schema.add_function(
                Function(
                    name='text_bm25',
                    input_field_names=['text'],
                    output_field_names=['sparse'],
                    function_type=FunctionType.BM25,
                )
            )
            indexes = self.client.prepare_index_params()
            indexes.add_index(
                field_name='dense', index_type='AUTOINDEX', metric_type='COSINE'
            )
            indexes.add_index(
                field_name='sparse',
                index_type='SPARSE_INVERTED_INDEX',
                metric_type='BM25',
            )
            self.client.create_collection(
                collection_name=self.collection,
                schema=schema,
                index_params=indexes,
                consistency_level='Strong',
            )
        # 对已有集合只校验而不覆盖，避免把错误配置当作新集合删除重建。
        fields = {
            f['name']: f
            for f in self.client.describe_collection(self.collection)['fields']
        }
        required = {
            'id',
            'dense',
            'text',
            'sparse',
            'category',
            'content_type',
            'product_category',
        }
        if (
            not required.issubset(fields)
            or int(fields['dense']['params']['dim']) != dim
        ):
            raise ValueError('Ch04集合schema或维度不匹配，未执行覆盖')
        self.client.load_collection(self.collection)
        self.ready_dim = dim
        return True

    async def upsert_chunk(self, row, vector):
        def work():
            with self.lock:
                self._ready(len(vector), True)
                # 主键和MySQL知识块一致；中断后重跑upsert不会新增重复向量。
                self.client.upsert(
                    self.collection,
                    [
                        {
                            'id': str(row['id']),
                            'dense': vector,
                            'text': embedding_text(row),
                            'category': row['category'],
                            'content_type': row['content_type'] or '',
                            'product_category': self.product_category(row),
                        }
                    ],
                )

        # 父类_call把同步SDK放进线程，并在取消时等待已发出的写操作收敛。
        await self._call(work)

    async def retrieve(self, text, vector, strategy, expression=''):
        def work():
            with self.lock:
                # BM25不需要请求嵌入；本章BGE-M3固定1024维。
                dim = len(vector) if vector is not None else 1024
                if not self._ready(dim):
                    return []
                kwargs = {'output_fields': ['id'], 'consistency_level': 'Strong'}
                # Strong保证刚写入的数据可立即读到，避免建库验收出现短暂空结果。
                if strategy == 'dense':
                    hits = self.client.search(
                        self.collection,
                        data=[vector],
                        anns_field='dense',
                        limit=50,
                        filter=expression,
                        search_params={'metric_type': 'COSINE'},
                        **kwargs,
                    )
                elif strategy == 'bm25':
                    hits = self.client.search(
                        self.collection,
                        data=[text],
                        anns_field='sparse',
                        limit=50,
                        filter=expression,
                        search_params={'metric_type': 'BM25'},
                        **kwargs,
                    )
                else:
                    # 同一谓词分别交给两条召回请求，不能等融合后才过滤。
                    reqs = [
                        AnnSearchRequest(
                            data=[vector],
                            anns_field='dense',
                            param={'metric_type': 'COSINE'},
                            limit=50,
                            expr=expression,
                        ),
                        AnnSearchRequest(
                            data=[text],
                            anns_field='sparse',
                            param={'metric_type': 'BM25'},
                            limit=50,
                            expr=expression,
                        ),
                    ]
                    hits = self.client.hybrid_search(
                        self.collection,
                        reqs=reqs,
                        ranker=RRFRanker(k=60),
                        limit=100,
                        **kwargs,
                    )
                # Milvus只返回主键/分数；正文留给Retriever回查MySQL权威原文。
                return [
                    {'id': str(h['id']), 'score': float(h['distance'])} for h in hits[0]
                ]

        return await self._call(work)

    async def search(self, vector, limit):
        """前章服务协议的dense兼容入口；新管线使用retrieve。"""
        hits = await self.retrieve('', vector, 'dense')
        return [(r['id'], r['score']) for r in hits[:limit]]

    async def close(self):
        await super().close()
        self.ready_dim = None
