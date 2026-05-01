"""HTTP REST API for the Ae105c trajectory optimization pipeline.

Run the server:
    python -m Python_Consolidated.api

Then visit http://localhost:8000/docs for browseable Swagger UI, or use
the Python client:

    from Python_Consolidated.api.client import Client
    cli = Client('http://localhost:8000')
    print(cli.list_results())
"""
__version__ = '0.1.0'
