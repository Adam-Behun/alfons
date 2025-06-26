from api.database import get_logs
import pandas as pd
import asyncio

async def export_to_csv():
    logs = await get_logs()
    df = pd.DataFrame(logs)
    df.to_csv("conversation_logs.csv", index=False)
    print("Logs exported to conversation_logs.csv")

if __name__ == "__main__":
    asyncio.run(export_to_csv())