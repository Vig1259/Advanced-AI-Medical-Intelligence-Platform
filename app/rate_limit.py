"""
Shared slowapi Limiter instance.

Defined in its own module (rather than directly in main.py) so router
modules can import and apply @limiter.limit(...) to individual endpoints
without creating a circular import with main.py (which also needs the
same instance to register it on app.state and wire up the exception
handler).
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)