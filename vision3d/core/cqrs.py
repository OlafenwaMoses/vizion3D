from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Type, TypeVar

TResult = TypeVar("TResult")


class Command(ABC, Generic[TResult]):
    pass


class Query(ABC, Generic[TResult]):
    pass


C = TypeVar("C", bound=Command)
Q = TypeVar("Q", bound=Query)


class CommandHandler(ABC, Generic[C, TResult]):
    @abstractmethod
    def handle(self, command: C) -> TResult:
        pass


class QueryHandler(ABC, Generic[Q, TResult]):
    @abstractmethod
    def handle(self, query: Q) -> TResult:
        pass


class CommandBus:
    def __init__(self, resolver: Callable[[Type[CommandHandler]], CommandHandler]):
        self._resolver = resolver
        self._registry: Dict[Type[Command], Type[CommandHandler]] = {}

    def register(self, command_type: Type[Command], handler_type: Type[CommandHandler]):
        self._registry[command_type] = handler_type

    def dispatch(self, command: Command) -> Any:
        handler_type = self._registry.get(type(command))
        if not handler_type:
            raise ValueError(f"No handler registered for command {type(command).__name__}")
        handler = self._resolver(handler_type)
        return handler.handle(command)


class QueryBus:
    def __init__(self, resolver: Callable[[Type[QueryHandler]], QueryHandler]):
        self._resolver = resolver
        self._registry: Dict[Type[Query], Type[QueryHandler]] = {}

    def register(self, query_type: Type[Query], handler_type: Type[QueryHandler]):
        self._registry[query_type] = handler_type

    def dispatch(self, query: Query) -> Any:
        handler_type = self._registry.get(type(query))
        if not handler_type:
            raise ValueError(f"No handler registered for query {type(query).__name__}")
        handler = self._resolver(handler_type)
        return handler.handle(query)
