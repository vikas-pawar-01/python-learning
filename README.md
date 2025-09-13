# Python

-------

## Python Must-Know Modules & Ecosystem (Q&A)

### 1. What are some must-know built-in Python modules?

👉

os → file system operations (paths, environment variables).

sys → interpreter-related tasks, command-line args.

datetime → working with dates/times.

json → parsing JSON.

logging → application logging.

re → regex operations.

subprocess → run shell commands.

math / decimal / fractions → numeric operations.

threading / multiprocessing → concurrency.

functools → higher-order functions (lru_cache, partial).

collections → Counter, defaultdict, deque, namedtuple.

itertools → infinite iterators, combinatorics (chain, permutations).

### 2. What are Python's most common third-party ecosystem libraries?

👉

Web frameworks: Django, Flask, FastAPI.

Data Science: NumPy, Pandas, Matplotlib, SciPy.

Machine Learning: scikit-learn, TensorFlow, PyTorch.

Databases: SQLAlchemy, psycopg2, PyMongo.

Async/Networking: aiohttp, requests, websockets.

DevOps: Docker SDK, Fabric, Ansible modules.

Testing: pytest, unittest, hypothesis.

### 3. Difference between import module vs from module import func vs import module as alias?

👉

import math → Must call with prefix: math.sqrt(16).

from math import sqrt → Direct use: sqrt(16).

import math as m → Alias: m.sqrt(16).

⚠️ Best practice: use import module (or alias) for readability, avoid polluting namespace.

### 4. What is __all__ in Python modules?

👉 __all__ defines what gets imported when using from module import *.
Example:

__all__ = ["func1", "ClassA"]

### 5. What’s the difference between Python’s Standard Library and Ecosystem?

👉

Standard Library = Built-in modules shipped with Python (os, sys, json).

Ecosystem = External community packages installed via pip (numpy, requests, sqlalchemy).

### 6. How do you manage Python environments and dependencies?

👉

venv → built-in virtual environments.

pip → package manager.

pip freeze > requirements.txt → save dependencies.

pip install -r requirements.txt → reinstall dependencies.

Advanced: poetry or pipenv for dependency management.

### 7. What are Python’s package distribution tools?

👉

setuptools → build & distribute packages.

wheel → binary distribution format.

twine → upload packages to PyPI.

poetry → modern dependency + packaging tool.

### 8. What are Python’s concurrency ecosystems?

👉

threading → I/O-bound concurrency.

multiprocessing → CPU-bound parallelism.

asyncio → async/await style, event loop based.

concurrent.futures → thread pool & process pool abstraction.

### 9. Explain WSGI, ASGI in Python ecosystem.

👉

WSGI = Web Server Gateway Interface, sync standard (Flask, Django).

ASGI = Asynchronous Server Gateway Interface, async standard (FastAPI, Starlette).

### 10. What testing frameworks are must-know in Python?

👉

unittest (built-in).

pytest (most popular, easy to use).

mock (patching in tests).

hypothesis (property-based testing).

🔹 Quick Summary Interview Table
Category            Must-Know Modules/Libs
System              os, sys, subprocess, logging
Data                json, csv, sqlite3, re
Iteration           itertools, collections
Math/Time           math, datetime, decimal
Concurrency         threading, multiprocessing, asyncio
Web                 Flask, Django, FastAPI
Data Science        NumPy, Pandas, Matplotlib
Testing             unittest, pytest, mock


### 11. What is the difference between requirements.txt and Pipfile?

👉

requirements.txt → simple list of dependencies (pip freeze > requirements.txt).

Pipfile → richer metadata (created by pipenv), includes dev/test/prod dependencies.

### 12. What is the difference between pip, pipenv, and poetry?

👉

pip → basic package installer.

pipenv → manages virtualenvs + dependencies (Pipfile).

poetry → modern dependency + packaging manager, simplifies publishing to PyPI.

### 13. What are Python wheels (.whl files)?

👉 Binary distribution format for Python packages.
Faster install than building from source.

### 14. What are __init__.py files used for?

👉

Marks a directory as a Python package.

Can also execute initialization code.

If empty, just signals “this is a package.”

### 15. What is the purpose of if __name__ == "__main__":?

👉 Ensures code runs only when executed directly, not when imported as a module.

### 16. What is sys.path and how does Python find modules?

👉

sys.path = list of directories Python searches for imports.

Order:

Current script’s dir

PYTHONPATH env var

Standard library

Site-packages

### 17. What is the Python Package Index (PyPI)?

👉 The official repository of Python packages (pip install requests fetches from PyPI).

### 18. What is the difference between absolute and relative imports?

👉

Absolute: from mypackage.module import func

Relative: from .module import func (dot = current package)

### 19. What is monkey patching?

👉 Dynamically modifying or extending a module/class at runtime.
Example:

import math
math.sqrt = lambda x: "hacked!"

### 20. How do you handle circular imports in Python?

👉

Reorganize code to avoid cycles.

Use local imports inside functions.

Use importlib for dynamic imports.

### 21. What is importlib used for?

👉 Module for importing modules programmatically at runtime.

import importlib
mod = importlib.import_module("math")
print(mod.sqrt(9))

### 22. What is the difference between venv and virtualenv?

👉

venv → built-in in Python 3.

virtualenv → older, external package, works in Python 2/3.

### 23. What are Python namespaces?

👉

Built-in → always available (len, id).

Global → module-level.

Local → inside functions/classes.

Enclosing → nested functions.

### 24. What is __pycache__ and .pyc files?

👉

Python caches compiled bytecode (.pyc) inside __pycache__.

Speeds up program startup.

### 25. How does import * work?

👉

Imports everything not starting with _.

Controlled by __all__ if defined.
⚠️ Not recommended (namespace pollution).

### 26. What’s the difference between import time and from time import sleep?

👉

import time → must call time.sleep(1).

from time import sleep → can call sleep(1) directly.

### 27. What are Python’s package distribution formats?

👉

Source Distribution: .tar.gz or .zip → build required.

Wheel (.whl): prebuilt, fast install.

### 28. What is Conda vs Pip?

👉

pip → Python-only packages (from PyPI).

conda → manages both Python and non-Python dependencies (C libs, ML tools).

### 29. What is site-packages in Python?

👉 The directory where third-party modules are installed.

### 30. What are entry points in Python packages?

👉 Define CLI commands for installed packages. Example in setup.py:

entry_points={
    "console_scripts": [
        "mytool=my_package.cli:main",
    ],
}

### 🔹 Summary of Key Ecosystem Concepts

Imports & Paths → import, sys.path, absolute/relative imports.

Package Management → pip, venv, poetry, pipenv.

Distribution → wheels, PyPI, setup.py, entry points.

Environment → virtualenvs, conda.

Caching → __pycache__, .pyc.

-------

### 🔹 Python Internals Interview Q&A

### 1. How does Python manage memory internally?

Python uses a private heap for memory allocation, managed by the Python Memory Manager.

Objects and data structures are stored in this heap.

Memory management involves:

Reference counting (primary mechanism)

Garbage collection (GC) via cyclic GC for unreachable cycles

### 2. What is reference counting in Python?

Each object keeps a count of references pointing to it (sys.getrefcount(obj)).

When the count drops to zero → object is destroyed and memory reclaimed.

Problem: cannot handle circular references (hence Python has GC).

### 3. How does Python’s garbage collector detect cycles?

Uses a generational garbage collector:

Objects are divided into generations (0, 1, 2).

Young objects are collected frequently, older objects less often.

Cycle detector identifies unreachable objects with references to each other.

### 4. What is the GIL (Global Interpreter Lock)?

A mutex in CPython ensuring only one thread executes Python bytecode at a time.

Prevents race conditions in memory management.

Limits multi-threaded CPU-bound performance but fine for I/O-bound tasks.

### 5. How are Python functions stored in memory?

Functions are first-class objects.

Internally represented by PyFunctionObject, which holds:

Reference to compiled bytecode (__code__)

Function globals

Default arguments

Closure variables

### 6. What is a Python frame object?

Every function call creates a frame object (contains local variables, instruction pointer, etc.).

Frames are linked via a call stack.

Accessible via inspect.currentframe().

### 7. What is the difference between is and == internally?

is → checks object identity (same memory address).

== → calls object’s __eq__() to compare values.

### 8. How does Python handle string interning?

Small strings and identifiers are interned for memory efficiency.

Example:

a = "hello"
b = "hello"
print(a is b)  # True (interned)


Large or dynamically created strings are not always interned.

### 9. How are Python integers stored internally?

Small integers (usually -5 to 256) are cached (singletons).

Larger integers are stored as arbitrary precision objects.

### 10. How does Python execute code internally?

Source code → AST (Abstract Syntax Tree) via parser.

AST → bytecode via compiler.

Bytecode → executed by the CPython virtual machine (interpreter loop).

### 11. What are Python descriptors?

Objects implementing __get__, __set__, or __delete__.

Power behind properties, methods, class variables.

Example:

class MyDescriptor:
    def __get__(self, obj, objtype=None):
        return "Got value"
class Test:
    x = MyDescriptor()
print(Test().x)  # "Got value"

### 12. What is MRO (Method Resolution Order) in Python?

Defines order of method lookup in multiple inheritance.

Uses C3 linearization.

Example:

class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass
print(D.mro())  # [D, B, C, A, object]

### 13. What is the difference between shallow copy and deep copy internally?

Shallow copy: creates a new object but references inner objects (copy.copy).

Deep copy: recursively copies all nested objects (copy.deepcopy).

### 14. What happens when you import a module?

Checks if module exists in sys.modules cache.

If not → finds module file using sys.meta_path finders.

Executes module code in a new namespace.

Stores result in sys.modules.

### 15. What is the difference between locals() and globals()?

globals() → returns dictionary of current global symbol table.

locals() → returns dictionary of local variables (may not always be writable).

### 16. What are Python slots (__slots__)?

Prevents creation of dynamic __dict__ per object → saves memory.

Example:

class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

### 17. How does Python implement decorators internally?

A decorator is just a function that takes another function and returns a new one.

At compile-time, @decorator syntax is applied.

### 18. What is the difference between CPython, PyPy, Jython?

CPython → default implementation in C.

PyPy → JIT-compiled for speed.

Jython → runs on JVM.

IronPython → runs on .NET CLR.

### 19. How does Python optimize performance internally?

String interning, small int caching.

Bytecode optimization.

Built-in C implementations for common operations (list.sort, dict).

PyPy JIT for runtime optimization.

### 20. What are Python metaclasses?

A class of a class.

Controls how classes are created.

Example:

class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

--------

### 🔹 FastAPI Internals Interview Q&A

### 1. How does FastAPI handle requests internally?

FastAPI is built on Starlette (ASGI framework) for request/response handling.

Workflow:

Uvicorn/Hypercorn (ASGI server) receives HTTP request.

ASGI passes it to Starlette → FastAPI routes.

FastAPI matches request with APIRouter.

Dependency injection system resolves required inputs.

Request handler executes → response returned through ASGI stack.

### 2. What role does ASGI play in FastAPI?

ASGI (Asynchronous Server Gateway Interface) = successor to WSGI.

Supports async I/O, WebSockets, background tasks.

FastAPI apps are just ASGI apps → run on ASGI servers like Uvicorn.

### 3. How does FastAPI implement dependency injection?

Each endpoint parameter can be annotated with Depends.

FastAPI builds a dependency graph at startup.

At runtime, it resolves dependencies recursively before calling the endpoint.

Example:

from fastapi import Depends

def get_db():
    return "DB Connection"

@app.get("/items/")
def read_items(db=Depends(get_db)):
    return {"db": db}


👉 get_db() runs first, result passed into read_items.

### 4. How does FastAPI validate and parse requests?

Uses Pydantic models for request/response validation.

Workflow:

Request body parsed from JSON → mapped to Pydantic model.

Validation occurs (type checking, constraints).

Invalid requests → automatic 422 Unprocessable Entity response.

### 5. What is the difference between Depends and BackgroundTasks?

Depends: Runs synchronously/async before endpoint executes (used for auth, DB session, etc.).

BackgroundTasks: Runs after response is sent (used for sending emails, logging).

### 6. How does FastAPI integrate with SQLAlchemy?

Typically via a session-per-request pattern:

Dependency provides a session.

Session is yielded and closed after request.

Example:

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

### 7. How does FastAPI handle concurrency?

Based on Python’s async/await with asyncio.

Endpoints can be defined as:

def → runs in a threadpool executor (for sync I/O, CPU-bound work).

async def → runs directly on the event loop (for async I/O).

### 8. What is the difference between def and async def endpoints in FastAPI?

def: executed in a thread pool → good for blocking I/O (DB drivers, file ops).

async def: non-blocking, runs on event loop → good for async libraries (HTTP calls, async DB).

### 9. How does FastAPI generate OpenAPI docs?

At startup, FastAPI inspects all routes, dependencies, and models.

Builds an OpenAPI schema (JSON spec).

Swagger UI & ReDoc render it at /docs and /redoc.

### 10. How does FastAPI support WebSockets?

Starlette provides WebSocket support.

FastAPI just wraps it with type hints.

Example:

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text("Hello")
    await ws.close()

### 11. How does FastAPI handle middleware?

Middleware is implemented as ASGI callables wrapping the app.

Each middleware receives (scope, receive, send).

Example:

@app.middleware("http")
async def add_header(request, call_next):
    response = await call_next(request)
    response.headers["X-App"] = "FastAPI"
    return response

### 12. What is the request lifecycle in FastAPI?

ASGI server receives HTTP request.

Middleware stack runs (logging, CORS, auth).

Route matching in APIRouter.

Dependency injection resolves inputs.

Request handler executes.

Response serialized (via JSONResponse, StreamingResponse, etc.).

Middleware post-processing runs.

Response returned to ASGI server.

### 13. How does FastAPI manage performance with async DB drivers?

Supports async DB drivers (like asyncpg for Postgres).

Uses async def + await to avoid blocking event loop.

SQLAlchemy 2.0 has async ORM support, fully compatible.

### 14. How does FastAPI differ from Flask internally?

Flask: WSGI, synchronous, request globals, no native async.

FastAPI: ASGI, async-first, dependency injection, auto validation/docs.

### 15. What is the difference between APIRouter and app routes?

APIRouter: lets you modularize endpoints (grouped routers).

Mounted into main FastAPI app via app.include_router().

### 16. How does FastAPI achieve high performance compared to Django/Flask?

Async support (ASGI).

Starlette (highly optimized).

Pydantic (Cython-based validation).

No blocking global locks like Django ORM’s transaction handling.

### 17. How does FastAPI handle streaming responses?

Uses StreamingResponse from Starlette.

Sends chunks of data instead of full content → memory-efficient.

Example:

from fastapi.responses import StreamingResponse

def iter_file():
    yield b"Hello"
    yield b" World"

@app.get("/stream")
def stream():
    return StreamingResponse(iter_file(), media_type="text/plain")

### 18. How does FastAPI handle exception handling?

Uses exception handlers registered with app.exception_handler().

Example:

@app.exception_handler(404)
async def not_found_handler(req, exc):
    return JSONResponse({"error": "Not found"}, status_code=404)

### 19. How does FastAPI scale horizontally?

Run multiple Uvicorn workers behind a process manager (Gunicorn).

Example:

gunicorn -k uvicorn.workers.UvicornWorker app:app -w 4

### 20. What are FastAPI’s limitations?

Async can be tricky with ORMs (not all DB drivers async).

GIL limits CPU-bound parallelism.

Startup overhead for dependency graph in very large apps.

-------

## Database 


### 🔹 1. psycopg (Driver level)

### What it is:
A low-level PostgreSQL driver for Python. It connects directly to the database using SQL strings.

You write raw SQL and execute it.

No abstraction — you manage queries, commits, and mappings yourself.

✅ Pros:

Lightweight, no extra layer.

Complete control over SQL (fast for complex queries).

Closer to PostgreSQL features.

❌ Cons:

More boilerplate.

You must handle schema mapping, relationships, SQL injection prevention manually.

Harder to maintain in large apps.

### Example (psycopg):

import psycopg2

conn = psycopg2.connect("dbname=mydb user=postgres password=secret host=localhost")
cur = conn.cursor()

cur.execute("SELECT id, name FROM users WHERE id = %s", (1,))
print(cur.fetchone())

cur.execute("INSERT INTO users (name, email) VALUES (%s, %s)", ("Alice", "alice@example.com"))
conn.commit()

cur.close()
conn.close()

### 🔹 2. SQLAlchemy Core (Query Builder)

What it is:
A higher-level abstraction built on top of DB drivers like psycopg.

Instead of writing raw SQL strings, you use Python objects to build SQL (select(), insert(), update(), etc.).

It still returns rows, but you don’t define Python classes for tables (unless you want to).

✅ Pros:

Prevents SQL injection (auto escapes).

Easier to build complex SQL programmatically.

DB-agnostic (works with PostgreSQL, MySQL, SQLite, etc.).

❌ Cons:

Still not object-oriented — you work with tables and rows, not Python objects.

Slightly more verbose than ORM for CRUD.

### Example (SQLAlchemy Core):

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select

engine = create_engine("postgresql+psycopg2://postgres:secret@localhost/mydb")
metadata = MetaData()

users = Table("users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("email", String)
)

# SELECT
with engine.connect() as conn:
    result = conn.execute(select(users.c.id, users.c.name).where(users.c.id == 1))
    print(result.fetchone())

# INSERT
with engine.connect() as conn:
    conn.execute(users.insert().values(name="Alice", email="alice@example.com"))
    conn.commit()

### 🔹 3. SQLAlchemy ORM (Object Relational Mapper)

What it is:
The ORM layer of SQLAlchemy that maps Python classes ↔ database tables.

Instead of dealing with rows, you work with Python objects.

Makes code more readable and maintainable for large apps.

✅ Pros:

Cleaner, object-oriented syntax.

Handles relationships (User.posts, Post.author).

Great for large projects (e.g., FastAPI, Flask, Django).

❌ Cons:

Slightly slower than Core/raw SQL (extra abstraction).

Complex queries sometimes require falling back to Core.

### Example (SQLAlchemy ORM):

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("postgresql+psycopg2://postgres:secret@localhost/mydb")
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

# SELECT
user = session.query(User).filter_by(id=1).first()
print(user.name)

# INSERT
new_user = User(name="Alice", email="alice@example.com")
session.add(new_user)
session.commit()

🔹 Summary Table
Feature                 psycopg (Driver)    SQLAlchemy Core         SQLAlchemy ORM
Level of Abstraction    Lowest (raw SQL)    Medium (query builder)  Highest (OOP, classes)
SQL Writing             Manual strings      Python API (build SQL)  Mostly hidden
Safety (SQL Injection)  Manual escaping     Safe by default         Safe by default
Ease of Use             Low                 Medium                  High
Best For                Scripts, raw queries Query-heavy apps       Large apps, ORMs (FastAPI, Flask)

### 👉 Rule of thumb:

psycopg = best for scripts, raw queries.

SQLAlchemy Core = best for apps with complex SQL but no need for OOP.

SQLAlchemy ORM = best for web apps (Flask, FastAPI) where you want object-oriented models.


### We’ll use two tables:

users(id, name, email)
posts(id, title, user_id)


We’ll demonstrate:

SELECT with LIMIT/OFFSET

LEFT JOIN (users → posts)

### 🔹 1. psycopg2 (Driver Level – raw SQL)
import psycopg2

conn = psycopg2.connect("dbname=mydb user=postgres password=secret host=localhost")
cur = conn.cursor()

# SELECT with LIMIT & OFFSET
cur.execute("SELECT id, name FROM users ORDER BY id LIMIT 5 OFFSET 5;")
print(cur.fetchall())

# LEFT JOIN (users with their posts, even if no posts exist)
cur.execute("""
    SELECT u.id, u.name, p.title
    FROM users u
    LEFT JOIN posts p ON u.id = p.user_id
    ORDER BY u.id;
""")
print(cur.fetchall())

cur.close()
conn.close()

### 🔹 2. SQLAlchemy Core (Query Builder)
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey, select

engine = create_engine("postgresql+psycopg2://postgres:secret@localhost/mydb")
metadata = MetaData()

users = Table("users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("email", String)
)

posts = Table("posts", metadata,
    Column("id", Integer, primary_key=True),
    Column("title", String),
    Column("user_id", Integer, ForeignKey("users.id"))
)

with engine.connect() as conn:
    # SELECT with LIMIT & OFFSET
    stmt = select(users.c.id, users.c.name).order_by(users.c.id).limit(5).offset(5)
    print(conn.execute(stmt).fetchall())

    # LEFT JOIN
    stmt = (
        select(users.c.id, users.c.name, posts.c.title)
        .select_from(users.outerjoin(posts, users.c.id == posts.c.user_id))
        .order_by(users.c.id)
    )
    print(conn.execute(stmt).fetchall())

### 🔹 3. SQLAlchemy ORM (Object-Oriented)
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

engine = create_engine("postgresql+psycopg2://postgres:secret@localhost/mydb")
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="posts")

# SELECT with LIMIT & OFFSET
users_page2 = session.query(User).order_by(User.id).offset(5).limit(5).all()
for u in users_page2:
    print(u.id, u.name)

# LEFT JOIN (Users with Posts)
results = (
    session.query(User.id, User.name, Post.title)
    .outerjoin(Post, User.id == Post.user_id)  # LEFT JOIN
    .order_by(User.id)
    .all()
)
for row in results:
    print(row)

🔹 Output Example (if we had 7 users, 2 posts)
[(6, 'User6'), (7, 'User7')]   # from LIMIT/OFFSET
[(1, 'Alice', 'Hello World'),
 (2, 'Bob', None)]             # LEFT JOIN (Bob has no posts → None)


✅ Key takeaways:

psycopg2 → write raw SQL.

Core → SQL builder style, still SQL-like but Pythonic.

ORM → object-oriented, works with Python classes.

-------

## Thread and Process

### 🔹 1. Threading Example

Threads share the same memory space, so they’re lightweight but limited by the Global Interpreter Lock (GIL) in Python. Good for I/O-bound tasks (network calls, file I/O, DB queries).

import threading
import time

def worker(name):
    print(f"Thread {name} starting")
    time.sleep(2)  # simulate work (I/O wait)
    print(f"Thread {name} finished")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All threads completed")


Output (runs concurrently, order may vary):

Thread 0 starting
Thread 1 starting
Thread 2 starting
Thread 0 finished
Thread 1 finished
Thread 2 finished
All threads completed

### 🔹 2. Multiprocessing Example

Processes run in separate memory spaces with their own Python interpreter → bypasses GIL, good for CPU-bound tasks (math, image processing, data crunching).

from multiprocessing import Process
import time
import os

def worker(name):
    print(f"Process {name} (PID={os.getpid()}) starting")
    time.sleep(2)  # simulate heavy CPU work
    print(f"Process {name} finished")

processes = []
for i in range(3):
    p = Process(target=worker, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

print("All processes completed")


Output (each process runs separately with its own PID):

Process 0 (PID=12345) starting
Process 1 (PID=12346) starting
Process 2 (PID=12347) starting
Process 0 finished
Process 1 finished
Process 2 finished
All processes completed

### 🔹 3. Key Differences Between Threading and Multiprocessing
Feature         Threading 🧵                     Multiprocessing ⚡
Memory          Shared memory (same process)    Separate memory (independent processes)
GIL impact      Affected (only 1 thread runs Python bytecode at a time) Not affected (separate interpreters)
Best for        I/O-bound tasks (network, DB, files)    CPU-bound tasks (math, data crunching)
Overhead        Low (lightweight)               Higher (process creation is expensive)
Communication   Via shared variables/locks      Via queues/pipes (need serialization)
Debugging       Easier                          Harder (multi-process debugging)

### ✅ Rule of thumb:

Use threading for waiting tasks (web requests, DB queries, file read/write).

Use multiprocessing for CPU-intensive tasks (image processing, ML training, big computations).

---------

## Unit Test

### 🔹 1. Basic Unit Test Example
#### calculator.py
def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero!")
    return a / b

#### test_calculator.py
import unittest
from calculator import add, divide

class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        with self.assertRaises(ValueError):
            divide(5, 0)

if __name__ == "__main__":
    unittest.main()


✅ This shows basic test structure.

### 🔹 2. Mock Example (using unittest.mock)

Suppose you have a function that calls an external API:

####  weather.py
import requests

def get_weather(city):
    response = requests.get(f"http://api.weather.com/{city}")
    return response.json()


You don’t want to hit the real API in unit tests. So you mock requests.get.

#### test_weather.py
import unittest
from unittest.mock import patch
from weather import get_weather

class TestWeather(unittest.TestCase):

    @patch("weather.requests.get")
    def test_get_weather(self, mock_get):
        mock_get.return_value.json.return_value = {"temp": 25, "city": "Paris"}

        result = get_weather("Paris")

        self.assertEqual(result["temp"], 25)
        self.assertEqual(result["city"], "Paris")
        mock_get.assert_called_once_with("http://api.weather.com/Paris")

if __name__ == "__main__":
    unittest.main()


✅ Here, mock_get replaces the actual requests.get.

### 🔹 3. Stub Example

A stub is a simplified function/object used instead of a real dependency. Unlike a mock, it doesn’t verify calls — it just returns predefined values.

#### db.py
class Database:
    def get_user(self, user_id):
        raise NotImplementedError("Real DB not implemented yet")


Stub version for testing:

#### test_db.py
import unittest
from db import Database

class StubDatabase(Database):
    def get_user(self, user_id):
        return {"id": user_id, "name": "Test User"}

class TestDatabase(unittest.TestCase):

    def test_get_user(self):
        db = StubDatabase()
        user = db.get_user(1)
        self.assertEqual(user["name"], "Test User")

if __name__ == "__main__":
    unittest.main()


✅ The stub provides a fixed response for testing.

### 🔹 Key Differences: Mock vs Stub

Stub → passive, provides canned responses (used for setup).

Mock → active, can assert how it was called (used for verification).

---------

## FAST API

### 1. What is FastAPI? What are its main advantages over frameworks like Flask or Django REST Framework?

Answer:

FastAPI is a modern, fast (high-performance) web framework for building APIs using Python 3.7+ that makes use of Python’s type hints.

It is built on top of Starlette (for ASGI, routing, middleware etc.) and Pydantic (for data validation and settings management).

Major advantages:

Automatic validation of request data (body, query params, path params, headers, cookies) via type hints + Pydantic

High performance owing to asynchronous support via ASGI, event loop, etc.

Auto-generated docs: automatic OpenAPI and JSON Schema, Swagger UI / ReDoc out of the box.

Dependency injection system to organize code cleanly, handle shared resources (DB sessions, settings), easier testing.

Standards-based: uses OpenAPI, JSON Schema etc.

Good for modern API design tasks (async, background tasks, WebSockets etc).

Compared to Flask:

Flask is synchronous by default (though can be async to some extent), less built-in validation, fewer built-in features.

You’d need to add a lot manually (input validation, docs, etc.).

Compared to Django / DRF:

Django is more opinionated, heavier, usually more suited for full MVC apps or with templating etc.

DRF provides a lot, but may have more overhead; FastAPI tends to be lighter, more performant for API-only services.

### 2. Describe FastAPI’s dependency injection system. How is it used, what problems does it solve?

Answer:

FastAPI uses a function parameter inspection mechanism via Depends(...), which allows you to declare dependencies in path operations, routers, or other dependencies.

A dependency is just a callable (sync or async) which can itself accept other dependencies, allowing chaining.

Uses cases:

Shared resources: e.g. a database session, cache client, settings (config).

Authentication/Authorization: define a dependency that ensures the user is authenticated, or has certain permissions.

Pre / post logic: e.g. verification, logging, rate limiting.

What problems it solves:

Keeps handlers (endpoints) cleaner and more focused on business logic.

Helps reuse code (you define a dependency once, use it in many routes).

Easier to inject test doubles or mocks for testing.

Better modularity, separation of concerns.

### 3. How do you handle asynchronous operations in FastAPI? When should endpoints be async vs sync?

Answer:

FastAPI supports both synchronous (regular def) and asynchronous (async def) endpoints.

Async endpoints are useful when you have I/O bound tasks: calling external APIs, database queries (via async drivers), file I/O, network operations etc.

Sync endpoints are fine for CPU-bound work or when using synchronous ORMs / libraries. But mixing sync I/O inside async def (that blocks) can degrade performance.

Need to ensure that libraries used are compatible: for example, using async drivers for databases (e.g. asyncpg, SQLAlchemy’s AsyncSession, etc.), or using asynchronous HTTP clients (like httpx), otherwise the async advantage is lost.

Also, there’s overhead with async; for simple tasks, sync may suffice. But in highly concurrent scenarios, async helps more.

### 4. How do you manage database connections and transactions in FastAPI, especially in async contexts?

Answer:

Use a database library that supports async. For example, SQLAlchemy 1.4+ has AsyncSession, or use an async ORM like Tortoise, or use libraries like Databases.

Create a dependency that yields a database session (async), and ensure proper cleanup: opening at the start of request, closing or rolling back on exceptions, committing when needed.

For transactions:

Use async with session.begin() or similar to group multiple operations under a single transaction.

On exceptions, ensure rollback.

Use connection pooling: make sure the driver / ORM is setting up pools, so you do not open/close connections on every request.

Be careful of stale sessions / sessions used across threads (if mixing sync/async), avoid race conditions.

### 5. What are background tasks in FastAPI, and when & how would you use them?

Answer:

FastAPI has support for background tasks via BackgroundTasks (from fastapi) and also events (startup/shutdown), or external task queues.

BackgroundTasks is good for lightweight, fire-and-forget tasks that run after the response is sent, e.g. sending notification emails, logging, cleanups etc.

For heavier tasks, or tasks that should survive process restarts, use external task queue systems (Celery, RQ, or something like that).

Usage example:

from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(message)

@app.post("/items/")
async def create_item(item: Item, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, f"Created item {item.id}")
    return item


Key is tasks should be non-blocking and not impact response time. Also ensure errors in background tasks are handled / logged / retried if necessary.

### 6. How do you handle error and exception handling in FastAPI? Custom exception handlers, validation errors etc.

Answer:

FastAPI provides automatic validation errors (via Pydantic) for request body, path/query params etc, and returns 422 responses on invalid input.

For custom errors / domain errors, you can define custom exceptions and register exception handlers using @app.exception_handler(...).

Example:

class ResourceNotFound(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(ResourceNotFound)
async def resource_not_found_handler(request: Request, exc: ResourceNotFound):
    return JSONResponse(
        status_code=404,
        content={"message": f"Resource {exc.name} not found"}
    )


Also middleware can help capturing unexpected errors, logging them etc.

Use proper HTTP status codes, consistent error response formats (so client side can handle uniformly).

### 7. How would you implement rate limiting in a FastAPI app?

Answer:

Several approaches:

Use a middleware component that intercepts requests and tracks count per client (IP / user) and time window.

Use external storage (Redis or in-memory, depending on scale) to maintain request counts.

Algorithms: fixed window, sliding window, token bucket, leaky bucket etc.

Example using Redis fixed/sliding window:

On each request, key = user_id or IP + endpoint.

Redis INCR on key, set expiry of key to window (e.g. 1 minute) if not exists.

If incremented count > limit, reject with 429 status.

Use libraries / plugins like slowapi (which wraps starlette / FastAPI) or custom.

Also allow whitelisting, customizing by route, scaling features.

### 8. How does FastAPI support file uploads and streaming responses? What are things to be careful about?

Answer:

File uploads via UploadFile and File from fastapi. UploadFile is preferred since it avoids loading the entire file into memory; it provides a file‐like async interface.

Example:

from fastapi import File, UploadFile

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    # or stream it, or save it


Streaming responses using StreamingResponse from starlette.responses for large data, so as not to load the full content in memory:

from fastapi.responses import StreamingResponse

def generate_large_data():
    for chunk in get_chunks():
        yield chunk

@app.get("/download/")
async def download():
    return StreamingResponse(generate_large_data(), media_type="application/octet-stream")


Things to watch out:

Memory usage: avoid reading entire large files into memory.

Timeouts: large uploads/downloads may require configuring server timeouts.

Security: validate file types, sizes; care with paths (avoid path traversal etc).

I/O blocking: if writing to disk, ensure you're using async compatible I/O if concurrency is required.

### 9. How do you write tests for FastAPI applications? What tools / practices?

Answer:

Use pytest as the test framework usually.

FastAPI provides TestClient (from starlette.testclient) for sync testing, and tools like httpx.AsyncClient for async tests.

Use fixtures to setup test DB (in-memory, or a test database schema), seed data, cleanup between tests.

Mock dependencies via dependency overrides (FastAPI allows overriding dependencies).

Test schemas / validation: test invalid inputs (missing fields, wrong types etc), test success responses.

Test error handlers, edge cases, authorization flows.

Use coverage tools to ensure what parts are tested.

If there are background tasks, test their invocation (could mock external systems).

For integration tests, test Docker-ized setup, if using containers.

### 10. How would you deploy a FastAPI application to production? What are important factors?

Answer:

Use ASGI servers like Uvicorn, or Uvicorn behind a process manager (e.g. Gunicorn with uvicorn workers).

Containerization using Docker is common.

Use reverse proxy (Nginx) for:

SSL / TLS termination

Serving static files

Load balancing

Monitoring & logging: capture logs, metrics (response times, error rates). Use tools like Prometheus, Grafana etc.

Environment configuration: secrets, environment variables, config management.

Scaling: multiple instances, using load balancers; stateless services where possible; handling sessions appropriately.

Database migrations: using tools like Alembic for SQLAlchemy etc. Setup migrations in CI/CD or entrypoint.

Security: secure communication (HTTPS), authentication, authorization, input sanitization, protecting against injection, CORS configuration.

Performance optimizations: caching, proper use of async, efficient serialization, maybe response compression.

Handling lifecycle events: graceful shutdown, startup events (e.g. event handlers) to initialize connections etc.

### 11. How would you optimize the performance of a FastAPI app?

Answer:

Use async endpoints, asynchronous database drivers so I/O does not block.

Use caching:

At endpoint level: e.g. cache responses (Redis or in-memory).

At database query level.

Use pagination, cursor queries instead of returning huge data sets.

Optimize serialization: avoid heavy nested models, large payloads.

Use compression (Gzip, Brotli) for responses.

Use connection pools.

Use profiling tools to identify bottlenecks.

Use more worker processes / uvicorn workers if CPU bound.

Offload heavier tasks to background or external services.

Use proper logging levels; avoid expensive computations in logging.

Use HTTP/2 where appropriate.

### 12. How do you handle authentication & authorization in FastAPI?

Answer:

Authentication can be done via JWT tokens, OAuth2 (FastAPI supports OAuth2 flows), or cookie-/session-based auth depending on application.

Use dependencies to enforce authentication, e.g.:

from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user


Authorization: once user identity is known, check roles / permissions. Could be via dependency again.

Use scopes with OAuth2, or custom roles.

Secure endpoints with appropriate status codes, security headers (CORS, HSTS etc).

Store secrets securely, use secure password hashing (bcrypt, argon2 etc).

Consider refresh tokens, token expiration etc.

### 13. How would you implement API versioning in FastAPI?

Answer:

Several approaches:

URL versioning: include version in path, e.g. /v1/items/, /v2/items/

Header versioning: custom request header indicating version

Query parameter versioning: e.g. ?version=1

Subdomain versioning (less common)

Combine versioning with routers:

from fastapi import APIRouter

router_v1 = APIRouter(prefix="/v1")
router_v2 = APIRouter(prefix="/v2")


Maintain backward compatibility; deprecate old versions carefully.

Versioning documentation automatically; ensure OpenAPI reflects multiple versions if needed.

### 14. What is middleware in FastAPI, and how is it used?

Answer:

Middleware is a layer which processes request/response globally, across all routes (or for certain routes).

FastAPI (via Starlette) has middleware classes that can intercept requests before they reach route handlers, and responses before they are sent.

Use cases:

Logging request/response time

Adding/removing headers

CORS, compression

Cross-cutting concerns like authentication, rate limiting (though often via dependencies)

Example:

from starlette.middleware.base import BaseHTTPMiddleware

class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Pre-processing
        response = await call_next(request)
        # Post-processing
        return response

app.add_middleware(MyMiddleware)

### 15. Explain how FastAPI handles validation & serialization of request and response data.

Answer:

Uses Pydantic models to define schemas for request bodies and response models. These models declare types, constraints (e.g. min/max length, regex etc).

For incoming data, FastAPI will:

Parse the JSON or form data, or path/query params.

Validate that the data conforms to type hints / Pydantic model; if not, return 422 Unprocessable Entity with detailed error messages.

For outgoing data, if you specify a response_model, FastAPI will also validate / serialize that, ensuring what is returned matches schema, and will filter out extra fields by default.

Pydantic also helps with data conversion: e.g. converting string to int (if type hint), parsing datetime.

### 16. Explain how dependency overrides are used for testing in FastAPI.

Answer:

FastAPI allows you to override dependencies via app.dependency_overrides.

For example, if you have a dependency that fetches a real database session, in tests you can override it to provide a mock session / test database.

Example:

def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


This helps isolate components / avoid hitting production resources, make tests repeatable.

### 17. How would you scale FastAPI applications to handle high traffic?

Answer:

Horizontal scaling: multiple instances behind a load balancer.

Use container orchestration (Kubernetes, ECS, etc.).

Ensure services are stateless where possible.

Use caching at different levels (CDN for static content; application cache for frequent queries; possibly caching proxies).

Use async extensively to avoid I/O blocking.

Optimize database queries (indexes, avoiding N+1, query optimization).

Use connection pooling.

Use proper orchestration for tasks: background queues, offload heavy processing.

Use monitoring tools to detect bottlenecks.

Use rate limiting, circuit breakers for resilience.

### 18. What are some security best practices when building APIs with FastAPI?

Answer:

Always validate and sanitize inputs (Pydantic helps).

Use HTTPS.

Secure authentication & authorization: strong password hashing, secure token generation, short lifetimes, proper revocation, refresh tokens.

Use CORS properly: allow only from trusted origins.

Prevent injection attacks (SQL injection, command injection etc).

Limit file upload size & validate file types.

Use security headers (e.g. Content Security Policy, X-Frame-Options, HSTS etc).

Avoid exposing sensitive information (stack traces, secret keys) in logs or error responses.

Keep dependencies up to date (vulnerabilities).

Rate limiting to prevent abuse.

### 19. How does FastAPI generate API documentation? How can you customize it?

Answer:

Using OpenAPI spec and JSON Schema, based on type hints and Pydantic models.

Out of the box, FastAPI provides two UIs: /docs (Swagger UI) and /redoc (ReDoc) which render the API documentation.

Customization:

You can set metadata: title, description, version in FastAPI(...) constructor.

You can supply summary and description for endpoints.

Use Pydantic’s Field(...) to add metadata for fields (e.g. description, example).

Use response_model, response_description, responses parameter to document different status codes.

Tags for grouping endpoints.

Security schemes (OAuth2, API keys etc) are reflected in docs.

### 20. What is ASGI and how is it relevant to FastAPI?

Answer:

ASGI = Asynchronous Server Gateway Interface. It's a standard interface between async Python web servers and frameworks, successor or complement to WSGI (Web Server Gateway Interface) for synchronous Python web apps.

FastAPI is ASGI based, i.e. it's designed to work with ASGI servers like Uvicorn, Hypercorn. This allows:

Native async/await support.

Handling multiple concurrent connections well (not being blocked on I/O).

WebSockets, HTTP/2 etc.

WSGI frameworks (Flask, Django traditional) are synchronous, which can limit performance in I/O bound workloads.

### 21. How would you do logging & monitoring for FastAPI applications?

Answer:

Logging:

Use Python’s logging module; configure formatters, handlers (console, files).

Use middleware or custom dependencies to log incoming requests, response status codes, timing.

Log errors with stack trace.

Correlation IDs in logs for tracing across requests / microservices.

Monitoring:

Metrics: request throughput, latency, error rates. Use tools like Prometheus, Grafana.

APM (Application Performance Monitoring) tools: e.g. New Relic, Datadog.

Health endpoints: a route to check health (DB connectivity etc).

Logs aggregation: use tools like ELK/ EFK stack, or services like Splunk.

Trace failures and slow endpoints; use profiling.

### 22. Discuss best practices for project structure / modularization in a large FastAPI project.

Answer:

Modular routers: group related endpoints under routers (e.g. user_router, admin_router etc).

Separate domains / layers: e.g. separate modules for models, schemas (Pydantic), database (ORM), services / business logic, controllers / routes, utils.

Config management: central place for configuration (via environment variables etc), maybe using Pydantic’s BaseSettings.

Dependency management: shared dependencies, avoid duplication.

Templates for error responses, consistent responses.

Use versioning (if needed) in project structure.

Static files / assets separate.

Tests organized by module.

Use code style (PEP8), linters, type checking (mypy) if feasible.

### 23. What are some pitfalls when mixing sync and async code in FastAPI?

Answer:

Blocking calls inside async endpoints: if you use a sync library that performs blocking I/O in an async def endpoint, that blocks the event loop, hurting concurrency and performance.

Avoid using CPU-intensive work inside async endpoints directly; instead offload to threads or process pools if needed.

Be careful with database drivers: using synchronous ORM in async code can block.

Thread safety: some libraries not safe in async contexts.

Deadlocks or waits if mixing sync and async incorrectly.

Overhead: in some cases async adds complexity; if you have simple, low-request apps maybe sync is simpler.

### 24. How do you design a scalable real-time system (e.g. chat) using FastAPI and WebSockets?

Answer:

Use FastAPI’s WebSocket support.

If many concurrent WebSocket connections, ensure the server(s) can scale: run multiple instances, manage connections possibly via a shared message broker (Redis pub/sub), or use WebSocket manager patterns.

For state (e.g. rooms, users), avoid storing in memory if you have multiple instances; use a shared store (Redis, etc).

Authentication over WebSocket: ensure the user is authenticated during handshake.

If messages are heavy or many clients, consider throttling, message queueing.

Consider message durability, ordered delivery, reconnection logic.

Also think about fallback (long polling) if needed.

### 25. How do you version, document, test, and deprecate public APIs (contract stability etc)?

Answer:

Versioning: via path, header etc as discussed.

Documentation: maintain clear API docs, examples, changelogs.

Testing: ensure new versions are tested, ensure backward compatibility tests (clients dependent on old version).

Deprecation strategy: mark old endpoints as deprecated in docs, support for some time, inform clients, provide migration path.

### 26. Advanced: Implement LRU Caching, or request deduplication, or sliding window rate limiting (algorithmic questions)

Answer Sketch:

LRU Cache: use collections.OrderedDict or functools.lru_cache for simple cases; for custom control, maintain a doubly-linked list + hash map.

Sliding window rate limiting: maintain timestamps in Redis sorted sets per user / key; on each request, remove timestamps outside the window, count the rest, check limit, add new timestamp.

### 27. Explain the Global Interpreter Lock (GIL) in Python and its implications for FastAPI applications.

Answer:

The GIL prevents multiple native threads from executing Python bytecode simultaneously in CPython.

Implications:

For CPU-bound workloads: multiple threads won’t help much; better to use multiple processes or offload CPU heavy work (e.g. via process pool, external services).

For I/O-bound work (which most web APIs are), async or threads help because while waiting for I/O, GIL is released in many cases (for native I/O).

So in FastAPI, heavy CPU tasks in an async def route may block the event loop.

### 28. How do you manage configuration and secrets in production for a FastAPI app?

Answer:

Use environment variables for configs (DB URLs, secret keys, etc).

Use Pydantic BaseSettings to manage them, load from .env or environment.

Keep secrets out of source control. Use vault services or secret managers (AWS Secrets Manager, Hashicorp Vault etc).

For different environments (dev/staging/prod), maintain separate config, possibly via separate environment variables or config files.

Encrypt or ensure restricted access to logs / configs.

----------

### 1. Core Python (30 Questions)

### Q1. What are Python’s key features?
A: Interpreted, dynamically typed, high-level, supports multiple paradigms (OOP, functional, procedural), huge standard library.

### Q2. Difference between is and == in Python?
A: is → identity (same object in memory), == → equality (same value).

### Q3. What are Python data types?
A: Numbers (int, float, complex), Sequence (list, tuple, range), Set, Dict, Boolean, String.

### Q4. Explain mutable vs immutable.
A: Mutable → can be modified (list, dict, set). Immutable → cannot be changed (tuple, str, int).

### Q5. What is Python’s GIL?
A: Global Interpreter Lock – ensures only one thread executes Python bytecode at a time. Limits true multithreading for CPU-bound tasks.

### Q6. What is a Python decorator?
A: A function that takes another function as input and extends/modifies its behavior without changing the source code.

### Q7. What is list comprehension?
A: A shorthand way to create lists: [x**2 for x in range(5)].

### Q8. Difference between tuple and list?
A: List is mutable, tuple is immutable, tuples can be used as dict keys.

### Q9. Explain Python’s garbage collection.
A: Uses reference counting + cyclic garbage collector for objects with circular references.

### Q10. What is with statement used for?
A: Context manager for resource management (auto-closes files, connections, etc.).

Q11. Difference between @staticmethod, @classmethod, and instance methods?
A:

Instance method → operates on object (self).

Class method → operates on class (cls).

Static method → no access to cls or self.

### Q12. What are Python modules and packages?
A: Module = single .py file. Package = collection of modules with __init__.py.

### Q13. Explain Python’s exception hierarchy.
A: BaseException → Exception → (ValueError, TypeError, KeyError, etc.).

### Q14. What is __init__.py used for?
A: Marks a directory as a Python package.

Q15. Explain *args and **kwargs.
A: *args → variable positional arguments, **kwargs → variable keyword arguments.

### Q16. What is Python’s __str__ vs __repr__?
A: __str__ → readable string for users. __repr__ → unambiguous string for developers.

### Q17. Explain shallow vs deep copy.
A: Shallow copy copies only references, deep copy recursively copies objects.

### Q18. What is Python’s yield?
A: Used in generators to return values lazily (one at a time).

### Q19. What is duck typing?
A: If an object behaves like expected, type is irrelevant. (“If it quacks like a duck…”).

### Q20. How to manage memory in Python?
A: Automatic garbage collection, weak references, del keyword.

### Q21. Difference between Python 2 and 3?
A: Print is function in 3, unicode default, integer division difference.

### Q22. What are Python’s built-in data structures?
A: List, tuple, dict, set, frozenset.

### Q23. What is __slots__ in Python?
A: Restricts object attributes to save memory.

### Q24. How is Python dynamically typed?
A: Type is checked at runtime, variables can change type.

### Q25. Explain Python’s import system.
A: Imports search in current dir, PYTHONPATH, system paths.

### Q26. What is a Python metaclass?
A: A class of classes, controls class creation.

### Q27. Explain Python’s @property decorator.
A: Defines getter/setter methods while accessing as attribute.

### Q28. What are Python’s built-in functions for iteration?
A: enumerate(), zip(), map(), filter(), reduce().

### Q29. What are Python’s magic methods?
A: Methods like __init__, __len__, __add__, __eq__ used for operator overloading and special behaviors.

### Q30. What is __call__ method?
A: Makes a class instance callable like a function.

2. Advanced Python (15 Questions)

### Q31. Difference between multiprocessing and multithreading?
A: Multiprocessing bypasses GIL, better for CPU-bound tasks. Multithreading is good for I/O-bound tasks.

### Q32. What is async/await in Python?
A: Used for asynchronous programming with asyncio.

### Q33. Explain coroutines.
A: Special functions that can pause execution (await) and resume later.

### Q34. What is Python’s __enter__ and __exit__?
A: Used in context managers (with statement).

### Q35. Explain weak references.
A: References that don’t increase ref count, useful for caching.

### Q36. Difference between deep copy and serialization?
A: Deep copy = memory duplication, serialization = convert to bytes (pickle/json).

### Q37. Explain monkey patching.
A: Dynamically modifying classes or modules at runtime.

### Q38. What are descriptors in Python?
A: Objects defining __get__, __set__, __delete__.

### Q39. Difference between asyncio and threading?
A: asyncio → single-threaded concurrency, cooperative. Threading → OS-level threads.

### Q40. What is event loop in asyncio?
A: Core loop that schedules and runs coroutines.

### Q41. How does Python handle memory leaks?
A: GC handles most, but leaks happen via circular references or global caches.

### Q42. Explain Python’s MRO (Method Resolution Order).
A: Defines order of method lookup in inheritance (C3 linearization).

### Q43. What is a Python singleton pattern?
A: Ensures only one instance of class (via metaclass or module-level var).

### Q44. How does asyncio.gather work?
A: Runs multiple coroutines concurrently, returns results as list.

### Q45. What are futures and tasks in asyncio?
A: Future = placeholder for result. Task = coroutine wrapped in Future.

---------

3. FastAPI (25 Questions)

### Q46. What is FastAPI?
A: Modern, async Python framework for APIs, built on Starlette & Pydantic.

### Q47. Why FastAPI over Flask/Django?
A: Faster, async support, type hints, automatic validation & docs (OpenAPI, Swagger, ReDoc).

### Q48. How does FastAPI use type hints?
A: For request validation, response models, and automatic docs.

### Q49. What are Pydantic models?
A: Data validation and parsing models using type hints.

Q50. How to define a GET endpoint in FastAPI?
A:

@app.get("/items/{id}")
def read_item(id: int): return {"id": id}


Q51. How to define a POST endpoint with body?
A:

@app.post("/items")
def create_item(item: Item): return item


Q52. How to add query parameters in FastAPI?
A:

@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10): return ...


### Q53. How to handle form data in FastAPI?
A: Use Form from fastapi.

### Q54. How to upload files in FastAPI?
A: Use File and UploadFile.

### Q55. Explain dependency injection in FastAPI.
A: Functions/classes that can be injected into routes for DB session, auth, etc.

### Q56. How to handle authentication in FastAPI?
A: OAuth2, JWT tokens, API keys via dependencies.

Q57. How to add middleware in FastAPI?
A:

@app.middleware("http")
async def log_requests(request, call_next): ...


### Q58. How to enable CORS in FastAPI?
A: Use CORSMiddleware.

### Q59. What is BackgroundTasks in FastAPI?
A: Run background jobs after request completes.

### Q60. How to serve static files in FastAPI?
A: StaticFiles from Starlette.

### Q61. How does FastAPI handle async vs sync functions?
A: Async → non-blocking, sync → runs in threadpool.

### Q62. What is response_model in FastAPI?
A: Ensures responses match a Pydantic model.

### Q63. How to validate query/path parameters?
A: Using Query and Path.

### Q64. What are routers in FastAPI?
A: Modular way to organize endpoints (APIRouter).

### Q65. How to document APIs in FastAPI?
A: Swagger UI (/docs), ReDoc (/redoc).

### Q66. How to add request/response headers?
A: Use Header, Response.

### Q67. How to connect DB in FastAPI?
A: Via SQLAlchemy or async drivers (databases, asyncpg).

### Q68. How to use caching in FastAPI?
A: Use fastapi-cache with Redis.

### Q69. How to test FastAPI apps?
A: With TestClient from Starlette.

### Q70. How to deploy FastAPI?
A: Using uvicorn, gunicorn, Docker, Kubernetes, AWS/GCP.

---------

4. Database & ORM (10 Questions)

### Q71. What is SQLAlchemy ORM?
A: ORM for Python to map classes to database tables.

### Q72. How to create models in SQLAlchemy?
A: Define classes with Column attributes.

### Q73. Difference between lazy and eager loading?
A: Lazy loads data on access, eager loads with query.

### Q74. How to handle migrations in FastAPI?
A: Using Alembic.

### Q75. What is connection pooling?
A: Reusing DB connections to improve performance.

### Q76. How to use async DB in FastAPI?
A: With async SQLAlchemy or databases library.

### Q77. How to prevent SQL injection in FastAPI?
A: Always use ORM or parameterized queries.

### Q78. How to paginate DB results?
A: Use limit and offset.

### Q79. How to implement transactions in SQLAlchemy?
A: Using session.begin() or async with session.begin().

### Q80. Difference between NoSQL and SQL?
A: SQL = structured, relations, ACID. NoSQL = flexible schema, horizontal scaling.

---------

5. Frontend + Full Stack Concepts (10 Questions)

### Q81. What is the role of frontend in full stack?
A: Handles UI/UX, communicates with backend APIs.

### Q82. How does React connect with FastAPI?
A: React fetches API endpoints exposed by FastAPI.

### Q83. What is CORS issue in frontend-backend integration?
A: Browser blocks cross-origin requests unless backend allows.

### Q84. How to optimize frontend API calls?
A: Use caching, pagination, batching.

### Q85. What is JWT and how frontend stores it?
A: JSON Web Token for auth. Stored in memory or HttpOnly cookie.

### Q86. Difference between REST and GraphQL?
A: REST = fixed endpoints, GraphQL = flexible queries.

### Q87. How to secure APIs in frontend integration?
A: Use HTTPS, tokens, rate-limiting.

### Q88. How to handle file uploads from frontend?
A: Use multipart form data.

### Q89. What is SPA (Single Page Application)?
A: Web app loads once, updates via API without reload.

### Q90. How to handle error responses in frontend?
A: Check HTTP codes, show user-friendly messages.

6. System Design + Deployment (10 Questions)

### Q91. How to scale FastAPI?
A: Horizontal scaling with multiple workers, load balancer, caching.

### Q92. Difference between monolith and microservices?
A: Monolith = single codebase, Microservices = independent deployable services.

### Q93. What is API rate limiting?
A: Restricting request count per user/IP.

### Q94. How to implement rate limiting in FastAPI?
A: Use Redis-based middlewares like slowapi.

### Q95. What is Docker and why use it?
A: Containerization for portability, consistency.

### Q96. How to run FastAPI with Docker?
A: Write Dockerfile with uvicorn entrypoint.

### Q97. How to deploy FastAPI on AWS?
A: Use ECS/EKS, Lambda, or EC2 with Docker.

### Q98. How to use Nginx with FastAPI?
A: As reverse proxy for SSL, load balancing.

### Q99. How to ensure API security in production?
A: HTTPS, JWT/OAuth2, input validation, rate limiting.

### Q100. How to monitor FastAPI in production?
A: Use Prometheus + Grafana, logging, APM tools.
--------

1. Advanced Python (25)

### Q101. What is Python’s __new__ method?
A: It controls object creation, called before __init__.

### Q102. Difference between deepcopy and pickle?
A: deepcopy = memory copy, pickle = serialization to bytes.

### Q103. How does Python’s memory model handle small integers and strings?
A: Interning: small ints (-5 to 256) and strings are cached for reuse.

### Q104. Explain contextvars in Python.
A: Provides context-local storage useful in async programming.

### Q105. What are Python’s weak references?
A: References that don’t increase reference count, used for caches.

### Q106. What is difference between copy.copy() and slicing [:] for lists?
A: Both shallow copy, but slicing works only on sequences.

### Q107. How does Python handle recursion limit?
A: Default ~1000, configurable via sys.setrecursionlimit.

### Q108. Explain Python’s __hash__ and __eq__.
A: Used to define custom hashing and equality in dict/set.

### Q109. Can Python functions be nested?
A: Yes, functions inside functions, closures capture outer variables.

### Q110. What is functools.lru_cache?
A: Caching decorator for memoization.

### Q111. What is difference between iterator and iterable?
A: Iterable = object with __iter__, Iterator = has __next__.

### Q112. How does Python’s __getattr__ differ from __getattribute__?
A: __getattr__ called only if attribute missing, __getattribute__ always called.

### Q113. How does Python implement switch-case?
A: No built-in, use dict mapping or match-case (Python 3.10+).

### Q114. What is Python’s abc module?
A: Abstract Base Classes for defining interfaces.

### Q115. Explain memoization.
A: Technique of caching results of expensive function calls.

### Q116. What is difference between shallow immutability and deep immutability?
A: Shallow = top-level immutable, deep = nested also immutable.

### Q117. How to implement singleton in Python?
A: Using metaclasses, decorators, or module-level global.

### Q118. Explain Python’s __del__.
A: Destructor called when object is garbage collected.

### Q119. What are slots in Python classes?
A: Limits attributes, saves memory, faster attribute access.

### Q120. Explain Python’s GIL workaround for CPU-bound tasks.
A: Use multiprocessing instead of threading.

### Q121. How does Python handle method overloading?
A: No true overloading, last defined method wins. Use default args.

### Q122. Explain any() vs all().
A: any() returns True if any element is true, all() if all are true.

### Q123. What is difference between id() and hash()?
A: id() = memory address, hash() = value hash used in dict/set.

### Q124. How to enforce immutability in custom classes?
A: Override __setattr__ and raise exception.

### Q125. What is Python’s dataclasses module?
A: Introduced in 3.7, simplifies boilerplate for classes with auto __init__, __repr__.

---------

### 2. FastAPI Advanced (25)

### Q126. How does FastAPI support WebSockets?
A: Using @app.websocket("/ws").

### Q127. How to stream responses in FastAPI?
A: Use StreamingResponse.

### Q128. Difference between FastAPI dependencies Depends() and middleware?
A: Depends() used per-route injection, middleware runs on all requests.

### Q129. How to handle exceptions globally in FastAPI?
A: Use @app.exception_handler.

### Q130. How to customize OpenAPI schema in FastAPI?
A: Override app.openapi() function.

### Q131. How to implement role-based access control in FastAPI?
A: Via dependencies checking JWT claims/roles.

### Q132. How to use async SQLAlchemy in FastAPI?
A: With async_engine and async_session.

### Q133. How to implement rate-limiting in FastAPI?
A: Use libraries like slowapi with Redis.

### Q134. How to implement GraphQL with FastAPI?
A: Use strawberry-graphql or graphene.

### Q135. How to version APIs in FastAPI?
A: Use routers with prefixes (/v1, /v2).

### Q136. How to serve HTML templates in FastAPI?
A: Use Jinja2Templates.

### Q137. How to do request validation for custom rules?
A: Use Pydantic validators (@validator).

### Q138. How to add Swagger UI authentication?
A: Add security schemes in OpenAPI.

### Q139. How to implement session-based authentication in FastAPI?
A: Use cookies with session IDs.

### Q140. How does FastAPI handle dependency injection lifecycle?
A: Supports per-request, global, or sub-dependency injection.

### Q141. How to log requests/responses in FastAPI?
A: Middleware with logging.

### Q142. How to enable HTTPS in FastAPI locally?
A: Use uvicorn --ssl-keyfile and --ssl-certfile.

### Q143. How to compress responses in FastAPI?
A: Use GZipMiddleware.

### Q144. How to use FastAPI with Celery?
A: For background distributed tasks with RabbitMQ/Redis.

### Q145. How to run multiple FastAPI apps in one project?
A: Use multiple APIRouters.

### Q146. How to use FastAPI with gRPC?
A: By running gRPC server separately and integrating with FastAPI.

### Q147. How to override dependency in testing?
A: Use app.dependency_overrides.

### Q148. How to do schema migration automatically in FastAPI?
A: Use Alembic autogenerate.

### Q149. How to add health check endpoint in FastAPI?
A: Simple /health route returning JSON.

### Q150. How to implement async background jobs without Celery?
A: Use FastAPI BackgroundTasks.

---------

### 3. Database & ORM (20)

### Q151. What is difference between ORM and raw SQL?
A: ORM abstracts queries, raw SQL gives full control.

### Q152. Explain N+1 query problem.
A: Too many queries due to lazy loading.

### Q153. How to prevent N+1 queries in SQLAlchemy?
A: Use joinedload.

### Q154. How to define relationships in SQLAlchemy?
A: Using relationship() and ForeignKey.

### Q155. Explain optimistic vs pessimistic locking.
A: Optimistic checks version numbers, pessimistic locks rows.

### Q156. How to implement soft delete?
A: Add is_deleted column instead of deleting row.

### Q157. What is database indexing?
A: Data structure that speeds up queries.

### Q158. Difference between unique and primary key?
A: Primary key = unique + not null, unique = only unique.

### Q159. How to do bulk inserts in SQLAlchemy?
A: Use session.bulk_save_objects.

### Q160. Explain connection pooling parameters.
A: pool_size, max_overflow, pool_timeout.

### Q161. What is database deadlock?
A: Two transactions waiting on each other’s locks.

### Q162. How to prevent SQL injection?
A: Always use ORM or parameterized queries.

### Q163. Difference between LEFT JOIN and INNER JOIN?
A: LEFT JOIN includes non-matching rows, INNER doesn’t.

### Q164. What is database sharding?
A: Splitting large DB into smaller distributed parts.

### Q165. Difference between denormalization and normalization?
A: Normalization = reduce redundancy, Denormalization = improve speed.

### Q166. How to implement caching for DB queries?
A: Use Redis/Memcached.

### Q167. Explain ACID properties.
A: Atomicity, Consistency, Isolation, Durability.

### Q168. Difference between SQLAlchemy ORM vs Core?
A: ORM = object-based, Core = lower-level SQL expression.

### Q169. How to log SQL queries in SQLAlchemy?
A: echo=True in engine.

### Q170. How to use UUID as primary key in SQLAlchemy?
A: Use UUID type or sqlalchemy.dialects.postgresql.UUID.

---------

4. System Design & DevOps (20)

### Q171. What is horizontal vs vertical scaling?
A: Horizontal = more machines, Vertical = more power.

### Q172. What is CAP theorem?
A: Consistency, Availability, Partition tolerance (pick 2).

### Q173. Explain microservices benefits.
A: Scalability, independent deployment, fault isolation.

### Q174. What is API gateway?
A: Entry point handling routing, auth, rate limiting.

### Q175. Difference between REST and gRPC?
A: REST = JSON/HTTP, gRPC = Protobuf/binary, faster.

### Q176. How to secure APIs?
A: HTTPS, JWT, rate limiting, input validation.

### Q177. How to handle logging in distributed systems?
A: Centralized logging with ELK stack.

### Q178. What is message queue?
A: Middleware (RabbitMQ, Kafka) for async communication.

### Q179. How to handle retries in API design?
A: Use exponential backoff.

### Q180. How to scale database in microservices?
A: Database per service, sharding, replication.

### Q181. Difference between monorepo and polyrepo?
A: Monorepo = one repo for all services, Polyrepo = separate repos.

### Q182. What is circuit breaker pattern?
A: Prevents cascading failures by cutting off failing services.

### Q183. Explain load balancing strategies.
A: Round robin, least connections, IP hash.

### Q184. What is blue-green deployment?
A: Two environments, switch traffic to new after testing.

### Q185. What is canary deployment?
A: Release to small group first before full rollout.

### Q186. What is Docker Compose?
A: Tool to manage multi-container apps.

### Q187. How to use environment variables in Docker?
A: Define in .env and pass to container.

### Q188. What is Helm in Kubernetes?
A: Package manager for Kubernetes.

### Q189. How to monitor FastAPI with Prometheus?
A: Add /metrics endpoint with prometheus_fastapi_instrumentator.

### Q190. How to implement CI/CD for FastAPI?
A: GitHub Actions, Jenkins, GitLab CI with Docker build & deploy.

5. Frontend + Full Stack Advanced (10)

### Q191. How does React fetch API data?
A: Using fetch or axios inside useEffect.

### Q192. What is hydration in React?
A: Converting static HTML to interactive React on client.

### Q193. How to secure JWT in frontend?
A: Store in HttpOnly cookies.

### Q194. What is CSRF and how to prevent?
A: Cross-site request forgery, prevent using tokens & SameSite cookies.

### Q195. What is SSR vs CSR?
A: Server-side rendering vs Client-side rendering.

### Q196. How to implement real-time updates in React + FastAPI?
A: WebSockets or SSE.

### Q197. What is Redux used for?
A: Centralized state management.

### Q198. How to lazy load components in React?
A: React.lazy and Suspense.

### Q199. What is tree shaking in frontend build?
A: Removing unused code in bundles.

### Q200. How to handle 401 unauthorized in frontend?
A: Intercept response, redirect to login.

---------

Core Python (Basics to Advanced)

### Q1. What are Python’s key features?

Interpreted, dynamically typed, high-level, object-oriented.

Rich standard libraries, cross-platform, supports multiple paradigms (OOP, FP, imperative).

### Q2. What is Python’s memory management model?

Uses reference counting + garbage collector (GC).

GC removes cycles using gc module.

Memory is managed in private heap space, not directly accessible to developers.

### Q3. Difference between deepcopy and shallow copy?

shallow copy: copies only references (changes in nested objects reflect).

deepcopy: creates entirely independent copy (no shared references).

### Q4. Explain Python’s GIL (Global Interpreter Lock).

Only one thread executes Python bytecode at a time (in CPython).

Affects CPU-bound tasks but not I/O-bound tasks.

Workarounds: multiprocessing, async, or Jython/PyPy.

### Q5. How is Python different from Java?

Python is dynamically typed, Java is statically typed.

Python runs slower but is more concise.

Python has GIL (thread limitation), Java has real multithreading.

Data Structures & Algorithms

### Q6. What’s the time complexity of Python’s list, dict, set?

list: append O(1), pop last O(1), insert/remove O(n), indexing O(1).

dict & set: average O(1) lookup, insert, delete (hashing).

### Q7. Difference between list, tuple, and set?

list: mutable, ordered, duplicates allowed.

tuple: immutable, ordered.

set: unordered, unique elements, mutable but unindexed.

### Q8. How do you implement a queue and stack in Python?

Stack: list.append() + list.pop().

Queue: collections.deque for O(1) append/pop from both ends.

### Q9. How to remove duplicates from a list while preserving order?

def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


### Q10. How does Python handle hashing in dictionaries?

Uses hash table + open addressing (collision resolution).

__hash__() + __eq__() determine uniqueness.

OOP & Design

### Q11. Explain method resolution order (MRO).

Defines order in which base classes are searched.

Uses C3 linearization (Class.__mro__).

### Q12. Difference between @staticmethod, @classmethod, @property?

@staticmethod: no self, utility functions.

@classmethod: has cls, modifies class state.

@property: makes method behave like an attribute (getter/setter).

### Q13. Multiple inheritance issue in Python?

Diamond problem.

Solved using MRO (C3 linearization).

### Q14. Explain duck typing.

Behavior determined by methods/attributes, not actual type.

Example: object is iterable if it implements __iter__, not if it inherits from Iterable.

### Q15. How does Python implement encapsulation?

Public: normal names.

Protected: _single_underscore.

Private: __double_underscore (name mangling).

Advanced Python

### Q16. What is a metaclass?

Class of a class.

Controls class creation (__new__, __init__).

Example: enforce singleton, add methods dynamically.

### Q17. Difference between is and ==?

is: identity check (same object).

==: equality check (same value).

### Q18. How are generators different from iterators?

Generator: function with yield, lazy evaluation, saves state.

Iterator: implements __iter__() and __next__().

### Q19. What are coroutines in Python?

Generalization of generators.

Defined with async def, awaitable with await.

Used for async I/O, concurrency.

### Q20. Explain context managers.

Manages resources with with statement.

Implements __enter__ and __exit__.

Concurrency & Parallelism

### Q21. When to use multithreading vs multiprocessing in Python?

Multithreading: I/O-bound tasks.

Multiprocessing: CPU-bound tasks (bypasses GIL).

### Q22. Difference between concurrent.futures.ThreadPoolExecutor and ProcessPoolExecutor?

ThreadPoolExecutor: uses threads, better for I/O.

ProcessPoolExecutor: uses processes, better for CPU-bound tasks.

### Q23. Explain async/await in Python.

Cooperative multitasking.

async defines coroutine, await pauses until result.

### Q24. How to handle race conditions in Python threads?

Use threading.Lock, RLock, or Queue.

### Q25. Explain difference between multiprocessing Queue and Manager?

Queue: inter-process communication.

Manager: provides shared objects (list, dict) across processes.

Modules & Libraries

### Q26. Explain difference between deepcopy in copy module and serialization (pickle).

deepcopy: in-memory recursive copy.

pickle: serializes objects to bytes for storage/networking.

### Q27. How do you handle large datasets in Python?

Use generators, yield, iterators.

Libraries: pandas, numpy, dask, polars.

### Q28. How does NumPy improve performance over lists?

Uses contiguous memory blocks, vectorized operations in C.

### Q29. Difference between @lru_cache and memoization?

@lru_cache (from functools) caches function results, supports LRU eviction.

Memoization is manual caching.

### Q30. Explain how Python logging works.

Configurable loggers, handlers, formatters.

Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.

Web & Frameworks

### Q31. Difference between Flask and Django?

Flask: microframework, lightweight, flexible.

Django: batteries-included, ORM, admin, security features.

### Q32. How do you secure a Django application?

CSRF tokens, SQL injection prevention (ORM), XSS protection (auto-escape), HTTPS, authentication middleware.

### Q33. Explain WSGI and ASGI.

WSGI: synchronous, Python web standard.

ASGI: asynchronous (supports websockets, async I/O).

### Q34. How to scale a Python web app?

Horizontal scaling with load balancers (NGINX, Gunicorn, uWSGI).

Use caching (Redis, Memcached).

Database replication & sharding.

### Q35. How to implement REST API in Python?

Use Flask/FastAPI/Django REST Framework.

Define endpoints, use JSON responses, handle status codes.

Testing & Debugging

### Q36. How do you test Python code?

unittest, pytest, nose.

Mocking with unittest.mock.

### Q37. Difference between unit test and integration test?

Unit test: tests single function/class.

Integration test: tests combined modules.

### Q38. How to profile performance of Python code?

cProfile, timeit, line_profiler, memory profiler.

### Q39. How do you debug Python applications?

pdb, logging, IDE debuggers, print (quick debugging).

### Q40. Explain TDD (Test-Driven Development) in Python.

Write test → run (fail) → implement → run (pass) → refactor.

System Design & Best Practices

### Q41. How do you design a scalable Python service?

Microservices architecture.

REST/GraphQL APIs.

Use async (FastAPI), caching, database sharding.

### Q42. How to optimize Python performance?

Use C extensions (Cython, NumPy).

Optimize loops with comprehension/map.

Use multiprocessing or async.

Profile and optimize bottlenecks.

### Q43. What is dependency injection in Python?

Passing dependencies (objects) from outside, not creating inside.

Increases testability, modularity.

### Q44. Explain SOLID principles in Python.

Applied via classes, abstraction, interfaces (via ABC).

### Q45. What design patterns have you used in Python?

Singleton, Factory, Observer, Decorator, Proxy.

Miscellaneous & Tricky

### Q46. Difference between @decorator and higher-order functions?

Both wrap functions, but @decorator is syntactic sugar for higher-order functions.

### Q47. What is monkey patching in Python?

Modifying behavior of library/class at runtime.

### Q48. Why is Python slow? How to make it faster?

Interpreter overhead, GIL.

Solutions: Cython, Numba, PyPy, multiprocessing.

### Q49. Explain with open("file.txt") as f:.

Uses context manager, automatically closes file.

### Q50. How do you deploy Python applications?

Package with Docker.

Deploy on AWS/GCP/Azure with Gunicorn, Nginx.

Use CI/CD pipeline (GitHub Actions, Jenkins).
------
25 Advanced Python Interview Questions

### Q1. Explain Python memory leaks and how to detect them.

Leaks occur via lingering references, globals, cycles not collected.

Detect with tracemalloc, gc.collect(), objgraph.

### Q2. What is the difference between __new__ and __init__?

__new__: creates a new object (called first, static method).

__init__: initializes object after creation.

### Q3. How do you implement immutability in Python classes?

Override __setattr__ to block modification.

Use namedtuple or dataclasses(frozen=True).

### Q4. Explain Python’s descriptor protocol.

Objects with __get__, __set__, __delete__.

Used in properties, ORM models.

### Q5. How do you handle circular imports?

Move imports inside functions.

Use importlib.

Restructure packages.

### Q6. What is the difference between exec() and eval()?

eval: evaluates expression, returns result.

exec: executes code block, returns None.

### Q7. Explain Python memory views.

memoryview: allows direct memory access without copying.

### Q8. What are Python slots?

__slots__ reduces memory by avoiding per-object __dict__.

### Q9. What is asyncio event loop?

Core of async. Manages tasks, coroutines, scheduling.

### Q10. Explain difference between synchronous, asynchronous, and concurrent programming.

Sync: one task at a time.

Async: tasks yield control, cooperative.

Concurrent: multiple tasks in progress (threads/processes).

### Q11. How to speed up Python numerical code?

Use NumPy, Numba, Cython.

Avoid Python loops (vectorization).

### Q12. Difference between pickling and JSON serialization?

Pickle: Python-specific, binary, unsafe across languages.

JSON: text-based, cross-language.

### Q13. What is monkey patching drawback?

Breaks maintainability, unexpected behavior.

### Q14. How do you implement custom iterators?

Define __iter__ and __next__.

### Q15. How do you manage dependencies in Python projects?

requirements.txt, pipenv, poetry.

### Q16. Difference between poetry and pipenv?

Both manage dependencies + virtualenvs.

Poetry adds packaging + publishing.

### Q17. What are Python type hints and benefits?

Optional annotations (def f(x: int) -> str).

Improves readability, tooling, static analysis (mypy).

### Q18. How do you optimize Python logging in production?

Use rotating file handlers, async logging, structured logs (JSON).

### Q19. Difference between multiprocessing and multithreading overhead?

Multiprocessing: higher memory, process spawn overhead.

Multithreading: lower memory, blocked by GIL.

### Q20. How to secure Python applications?

Input validation, escaping.

Use venv to isolate dependencies.

Avoid eval/exec.

### Q21. What’s the difference between dataclasses and namedtuple?

dataclasses: mutable, more flexible, default values.

namedtuple: immutable, memory efficient.

### Q22. Explain dependency injection in Python with example.

Pass dependencies as parameters instead of creating inside.

class Service:
    def __init__(self, db): self.db = db


### Q23. What are weak references in Python?

Do not increase refcount. Used in caches (weakref module).

### Q24. Explain caching strategies in Python applications.

In-memory (functools.lru_cache), Redis, file-based.

### Q25. How do you build a Python CLI tool?

argparse, click, typer.
------
🔹 25 FastAPI Interview Questions

### Q26. What is FastAPI and why is it faster than Flask/Django?

Based on Starlette (ASGI) + Pydantic.

Async-first, uses uvicorn/gunicorn.

### Q27. How does FastAPI handle validation?

Uses Pydantic models for automatic request/response validation.

### Q28. Explain request handling lifecycle in FastAPI.

ASGI server receives request.

Routes matched.

Dependencies resolved.

Request validated via Pydantic.

Response returned.

### Q29. What are dependencies in FastAPI?

Functions/classes injected into endpoints (Depends()).

Used for DB sessions, auth, business logic.

### Q30. How does FastAPI handle async?

Endpoints can be async def.

Runs on event loop via Uvicorn.

### Q31. What is difference between Sync and Async routes in FastAPI?

Sync: executed in thread pool (blocking).

Async: non-blocking, faster for I/O.

### Q32. How to integrate SQLAlchemy with FastAPI?

Create DB session per request (dependency).

Use ORM models mapped to tables.

### Q33. What is the role of Pydantic in FastAPI?

Validates input/output data.

Provides type hints, JSON schema.

### Q34. How do you handle authentication in FastAPI?

OAuth2 with JWT.

Dependency injection for user retrieval.

### Q35. How to secure FastAPI APIs?

HTTPS, OAuth2/JWT, CORS middleware.

Rate limiting (external libs).

### Q36. Explain FastAPI middleware.

Functions that run before/after requests.

Example: logging, auth, CORS.

### Q37. How to implement file upload/download in FastAPI?

Upload: File, UploadFile.

Download: FileResponse.

### Q38. How to serve static files in FastAPI?

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")


### Q39. How do you handle background tasks in FastAPI?

BackgroundTasks dependency runs after response.

### Q40. How to run FastAPI in production?

uvicorn app:app --workers 4 --host 0.0.0.0.

Reverse proxy with Nginx/Gunicorn.

### Q41. How does FastAPI generate OpenAPI docs?

Automatic via Pydantic + type hints.

/docs (Swagger UI), /redoc (ReDoc).

### Q42. How to handle rate limiting in FastAPI?

Use middleware like slowapi.

### Q43. How do you implement WebSockets in FastAPI?

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text("Hello")


### Q44. How does dependency override work in FastAPI testing?

Use app.dependency_overrides to replace dependencies with mocks.

### Q45. How to test FastAPI endpoints?

TestClient from starlette.testclient.

Pytest fixtures for DB mocking.

### Q46. How do you deploy FastAPI with Docker?

Base image: python:3.11-slim.

Install dependencies, run uvicorn.

### Q47. How to handle DB transactions in FastAPI?

Use dependency with yield to manage session commit/rollback.

### Q48. How to implement pagination in FastAPI?

Query parameters (limit, offset) with Pydantic validation.

### Q49. Difference between Flask + FastAPI request handling?

Flask: WSGI, sync only.

FastAPI: ASGI, async/sync both.

### Q50. How do you monitor FastAPI in production?

Middleware logging.

Prometheus + Grafana.

APM (New Relic, Datadog).

---------

### 1. Python Core (20 Questions)

### What are Python data types you use most frequently?
Answer: int, float, str, list, dict, tuple, set, bool, NoneType. Each serves a purpose: lists for ordered collections, dicts for key‑value mapping, tuples for immutability.

### Difference between mutable and immutable objects?
Answer: Mutable objects (list, dict, set) can be changed in place. Immutable (str, tuple, frozenset) cannot; changes create a new object. This affects performance and hashing.

### What is a Python virtual environment and why use it?
Answer: An isolated environment to manage dependencies per project, avoiding version conflicts. Tools: venv, virtualenv, pipenv.

### Explain Python's GIL.
Answer: The Global Interpreter Lock allows only one thread to execute Python bytecode at a time. Concurrency is achieved with multiprocessing or async IO for CPU‑bound tasks.

### Difference between deepcopy and copy.
Answer: copy.copy() makes a shallow copy (nested objects are shared). copy.deepcopy() recursively copies all objects.

### Explain Python decorators.
Answer: Functions that wrap other functions to modify behavior. Common use cases: logging, authentication, caching.

### How do you manage memory in Python?
Answer: Python uses reference counting + garbage collector for cyclic references. You can force GC using gc.collect().

Explain *args and **kwargs.
Answer: *args collects positional args as tuple, **kwargs collects keyword args as dict. Used for flexible APIs.

### What is context manager?
Answer: Object implementing __enter__ and __exit__, allowing resource management using with statement.

### Difference between is and ==.
Answer: is checks identity (same object in memory), == checks value equality.

### How do you handle exceptions gracefully?
Answer: Use try/except/finally, log errors, raise custom exceptions. Keep except blocks specific.

### Explain list comprehensions and generator expressions.
Answer: List comprehension builds list eagerly; generator expression yields items lazily for efficiency.

### How to use dataclasses?
Answer: Use @dataclass decorator to auto‑generate init, repr, eq. Good for lightweight data containers.

### What are Python descriptors?
Answer: Objects defining __get__, __set__, __delete__ methods, used in property, ORM field definitions.

### Explain MRO (Method Resolution Order).
Answer: Defines order of class hierarchy lookup. Follows C3 linearization.

### Difference between classmethod and staticmethod.
Answer: classmethod receives class as first argument, used for alternate constructors. staticmethod is a plain function inside class namespace.

### What are Python typing hints?
Answer: Optional annotations for function signatures and variables to improve readability and tooling.

### What are Python contextvars?
Answer: Thread/Task local storage for async applications.

### Explain Python's __slots__.
Answer: Defines fixed set of attributes, reducing memory footprint by disabling dynamic __dict__.

### How to optimize Python performance?
Answer: Use built‑in functions, avoid unnecessary loops, leverage NumPy, caching, concurrency, and profiling tools (cProfile).

### 2. FastAPI Core (20 Questions)

### What is FastAPI?
Answer: A modern async Python web framework built on Starlette (ASGI) + Pydantic. Provides type‑driven validation, auto OpenAPI docs, async support, and great performance.

### How does dependency injection work in FastAPI?
Answer: Using Depends(). FastAPI resolves function parameters automatically, allowing shared logic (DB sessions, auth) to be injected.

### Explain sync vs async routes.
Answer: def routes run in threadpool. async def run on event loop. Use async for I/O bound tasks.

### What is response_model?
Answer: A Pydantic model used to serialize and validate output, filtering unwanted fields and generating documentation.

### How to add middleware?
Answer: Use app.add_middleware with Starlette’s BaseHTTPMiddleware.

### Explain startup/shutdown events.
Answer: Use @app.on_event("startup") and @app.on_event("shutdown") to initialize and cleanup resources.

### How to return streaming responses?
Answer: Use StreamingResponse with a generator or async iterator.

### Explain BackgroundTasks.
Answer: Allows execution of lightweight tasks after returning response.

### How does FastAPI auto‑generate docs?
Answer: Uses type hints, Pydantic models, and route definitions to build OpenAPI schema.

### What is APIRouter?
Answer: A way to modularize routes, group endpoints, set prefixes, tags, and shared dependencies.

### How to use security dependencies?
Answer: Use OAuth2PasswordBearer or custom dependencies with Depends().

### How to enable CORS?
Answer: Use CORSMiddleware with allowed origins, methods, headers.

### Explain file upload handling.
Answer: Use UploadFile and File() to handle streamed uploads without reading entire file into memory.

### How to mount static files?
Answer: app.mount("/static", StaticFiles(directory="static"))

### How to override dependencies for tests?
Answer: Set app.dependency_overrides[dep] = override_func.

### How to implement pagination?
Answer: Use query params (limit, offset) or cursor-based pagination. Return metadata.

### How to customize OpenAPI schema?
Answer: Override app.openapi method and return modified schema.

### How to handle custom exceptions?
Answer: Register handler with @app.exception_handler(MyException).

### When to choose FastAPI over Flask?
Answer: When async, type hints, auto docs, and high performance are required.

### 3. Async & Concurrency (15 Questions)

### What is ASGI?
Answer: Async Server Gateway Interface, supports async frameworks, WebSockets, background tasks.

### Explain concurrency vs parallelism.
Answer: Concurrency = task switching, parallelism = tasks running truly simultaneously.

### What happens if you block inside async def?
Answer: Event loop freezes. Use threadpool (run_in_executor) for blocking calls.

### What is asyncio event loop?
Answer: Core of async system scheduling coroutines.

### How to run CPU‑bound code safely?
Answer: Offload to process pool or background worker.

### Explain asyncio.gather.
Answer: Runs coroutines concurrently and aggregates results.

### Explain backpressure.
Answer: Mechanism to slow producers when consumers are overwhelmed.

### What are asyncio.Lock and Semaphore used for?
Answer: Protect shared resources, control concurrency.

### How to cancel async tasks?
Answer: Call task.cancel() and handle CancelledError.

### How to set timeouts in async code?
Answer: Use asyncio.wait_for(coro, timeout).

### How to debug async code?
Answer: Enable PYTHONASYNCIODEBUG=1, use logging, inspect tasks.

### Explain cooperative multitasking.
Answer: Coroutines yield control when awaiting I/O.

### How does Uvicorn handle concurrency?
Answer: Runs event loop (uvloop by default) and handles requests concurrently.

### How to handle WebSockets concurrently?
Answer: Use connection manager, broadcast with tasks.

### How to avoid race conditions?
Answer: Lock shared state, prefer stateless design.

### 4. Database & ORM (20 Questions)

### How to set up SQLAlchemy with FastAPI?
Answer: Create engine, session maker, Base class, yield session per request.

### Sync vs async SQLAlchemy?
Answer: Async uses async engine + AsyncSession, requires await. Sync uses threadpool.

### How to run migrations?
Answer: Use Alembic with revision --autogenerate and upgrade head.

### How to manage transactions?
Answer: Use session.begin() context manager, rollback on exception.

### What is eager loading?
Answer: Preload relationships to avoid N+1 queries.

### How to do bulk insert?
Answer: Use session.bulk_save_objects or insert().

### How to serialize ORM models?
Answer: Pydantic models with from_orm=True.

### How to handle DB errors?
Answer: Catch IntegrityError, rollback, return error response.

### What is connection pooling?
Answer: Maintains reusable DB connections for performance.

### How to do multi-tenancy?
Answer: Separate schemas or add tenant_id column with filters.

### How to seed data?
Answer: Run data insert script at startup or via migration.

### How to handle read replicas?
Answer: Route reads to replica engine, writes to primary.

### How to test with database?
Answer: Use a test DB, rollback transactions after each test.

### How to profile slow queries?
Answer: Enable echo, EXPLAIN queries, use monitoring tools.

### How to implement soft delete?
Answer: Add is_deleted column, filter in queries.

### How to handle connection retries?
Answer: Use pool_pre_ping=True, retry logic with Tenacity.

### How to enforce integrity at app level?
Answer: Validate with Pydantic, apply unique constraints.

### How to stream large query results?
Answer: Use yield_per() or chunked reads.

### 5. Security & Auth (15 Questions)

### How to hash passwords securely?
Answer: Use passlib with bcrypt/argon2.

### How to implement JWT auth?
Answer: Create access tokens with PyJWT, validate in dependency.

### How to refresh tokens?
Answer: Use refresh tokens with longer expiry, issue new access tokens.

### How to protect routes?
Answer: Use dependency that checks current user, raise 401 if invalid.

### Explain OAuth2PasswordBearer.
Answer: Extracts bearer token from Authorization header.

### How to implement RBAC?
Answer: Store user roles, check permissions in dependency.

-------
