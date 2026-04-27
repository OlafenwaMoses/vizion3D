from clean_ioc import Container

from .cqrs import CommandBus, QueryBus

# Core IoC container instance
container = Container()


def resolve_handler(handler_type):
    return container.resolve(handler_type)


command_bus = CommandBus(resolver=resolve_handler)
query_bus = QueryBus(resolver=resolve_handler)

container.register(CommandBus, instance=command_bus)
container.register(QueryBus, instance=query_bus)


def register_command_handler(command_type, handler_type):
    container.register(handler_type)
    command_bus.register(command_type, handler_type)


def register_query_handler(query_type, handler_type):
    container.register(handler_type)
    query_bus.register(query_type, handler_type)
