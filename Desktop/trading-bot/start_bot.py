import asyncio
from run_adaptive_intelligent_bot import AdaptiveIntelligentBot

async def main():
    bot = AdaptiveIntelligentBot()
    await bot.run_adaptive_intelligent_bot()

if __name__ == "__main__":
    asyncio.run(main())