import asyncio
import modal

async def main():
    app = await modal.App.lookup.aio("gpunity-sandbox", create_if_missing=True)
    sb = await modal.Sandbox.create.aio("bash", "-lc", "sleep 10", app=app)
    try:
        items = await sb.exec.aio("ls", "/tmp")
    except AttributeError:
        # modal < 1.1 doesn't have exec maybe
        pass
    
    try:
        items = []
        async for item in sb.ls.aio("/tmp"):
            items.append(item)
        print("Async generator ls:", items)
    except Exception as e:
        print("Error with async for:", e)

    try:
        items = await sb.ls.aio("/tmp")
        print("Await ls:", items)
    except Exception as e:
        print("Error with await ls:", e)

if __name__ == "__main__":
    asyncio.run(main())
