import asyncio
import modal
import time

async def main():
    image = modal.Image.debian_slim()
    app = await modal.App.lookup.aio("test-sandbox", create_if_missing=True)
    
    print("Creating sandbox...")
    sb = await modal.Sandbox.create.aio(
        "bash", "-c", "echo hello > /tmp/hello.txt; sleep 60",
        app=app,
        image=image
    )
    print("Sandbox created!")
    
    start = time.monotonic()
    while True:
        try:
            ls_res = await sb.ls.aio("/tmp")
            print(f"/tmp ls: {ls_res}")
            break
        except Exception as e:
            print(f"Exception calling ls: {e}")
            await asyncio.sleep(1)

        if time.monotonic() - start > 10:
            break

asyncio.run(main())
