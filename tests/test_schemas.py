import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.extract import AfterSalesTicket, ExtractRequest


def test_missing_order_serializes_as_null():
    ticket = AfterSalesTicket(request_type="退款", expected_solution="退还货款")
    assert ticket.model_dump() == {
        "order_id": None, "request_type": "退款", "expected_solution": "退还货款"
    }


@pytest.mark.parametrize("kind", ["退款", "换货", "维修", "投诉", "其他"])
def test_request_types(kind):
    assert AfterSalesTicket(request_type=kind, expected_solution="处理问题").request_type == kind


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        AfterSalesTicket(request_type="退换", expected_solution="处理问题")


@pytest.mark.parametrize("data", [
    {"session_id": "s", "message": " "},
    {"session_id": " ", "message": "你好"},
])
def test_empty_chat_rejected(data):
    with pytest.raises(ValidationError):
        ChatRequest(**data)


def test_blank_extract_rejected():
    with pytest.raises(ValidationError):
        ExtractRequest(text="\n ")
