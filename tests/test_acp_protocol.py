from densify.acp.protocol import parse_jsonrpc_message, request_from_message, response


def test_parse_jsonrpc_message():
    message = parse_jsonrpc_message('{"jsonrpc":"2.0","id":1,"method":"initialize"}')

    assert message["method"] == "initialize"


def test_request_from_message_defaults_params_to_empty_dict():
    request = request_from_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert request is not None
    assert request.id == 1
    assert request.method == "initialize"
    assert request.params == {}


def test_response():
    assert response(1, {"ok": True}) == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

