import asyncio
import modal

async def main():
    print("Checking app lookup...")
    app = await modal.App.lookup.aio("gpunity-sandbox", create_if_missing=True)
    print("App lookup done.")
    print("Creating sandbox...")
    sb = await modal.Sandbox.create.aio(
        "bash", "-lc", "echo Hello",
        app=app,
    )
    print("Sandbox created", sb.object_id)
    await sb.wait.aio()
    print("Done waiting.")
    
if __name__ == "__main__":
    asyncio.run(main())
