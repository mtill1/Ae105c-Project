"""`python -m Python_Consolidated.api` — launch the API server."""
import argparse
import uvicorn


def main():
    p = argparse.ArgumentParser(prog='python -m Python_Consolidated.api',
                                  description='Launch the Ae105c API server')
    p.add_argument('--host', default='127.0.0.1',
                   help='Bind address (default: 127.0.0.1, use 0.0.0.0 for LAN)')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--reload', action='store_true',
                   help='Auto-reload on code changes (dev only)')
    p.add_argument('--workers', type=int, default=1,
                   help='Worker processes (jobs.db is per-process, so keep at 1)')
    args = p.parse_args()

    print(f'\nAe105c API → http://{args.host}:{args.port}')
    print(f'Swagger UI → http://{args.host}:{args.port}/docs')
    print(f'Health     → http://{args.host}:{args.port}/health\n')

    uvicorn.run(
        'Python_Consolidated.api.server:app',
        host=args.host, port=args.port,
        reload=args.reload, workers=args.workers,
    )


if __name__ == '__main__':
    main()
