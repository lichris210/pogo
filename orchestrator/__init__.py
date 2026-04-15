"""POGO v2 orchestrator — stateful conversation manager.

Routes user messages through the multi-agent pipeline using a
DynamoDB-backed session state machine.

Entry point::

    from orchestrator.orchestrator import handle_message
"""
