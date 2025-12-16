
import asyncio
import signal
import sys
import logging
from logging.handlers import RotatingFileHandler

from config import config
from bot import get_bot

def setup_logging():
    config.logging.ensure_directory()
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.logging.level.upper()))
    
    file_handler = RotatingFileHandler(
        config.logging.file_path,
        maxBytes=config.logging.max_size_mb * 1024 * 1024,
        backupCount=config.logging.backup_count
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

async def main():
    logger = setup_logging()
    
    config.initialize()
    
    issues = config.validate()
    for issue in issues:
        logger.info(issue)
    
    logger.info("=" * 50)
    logger.info("Solana Trading Bot Starting")
    logger.info("=" * 50)
    logger.info(f"Network: {config.solana.network}")
    logger.info(f"Paper Trading: {config.trading.paper_trading}")
    logger.info(f"Database: {config.database.path}")
    
    bot = get_bot()
    
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    try:
        await bot.start()
        
        while bot.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await bot.stop()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
