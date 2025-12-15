import inspect
from chainlit import Message
print(Message.update)
print(inspect.signature(Message.update))
