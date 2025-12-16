import asyncio
import signal
import sys
import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Callable, Any
from datetime import datetime
from pathlib import Path
import traceback
import atexit
import gc

from config import config
from bot import TradingBot, get_bot
from database import Database
from audit_secure import AuditLogger


class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.shutdown_callbacks: list[Callable] = []
        self._shutting_down = False
        
    def register_callback(self, callback: Callable) -> None:
        self.shutdown_callbacks.append(callback)
        
    async def trigger_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.shutdown_event.set()
        
        for callback in reversed(self.shutdown_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logging.error(f"Shutdown callback error: {e}")


class ApplicationLogger:
    def __init__(self, config):
        self.config = config
        self.logger = None
        
    def setup(self) -> logging.Logger:
        log_dir = Path(self.config.logging.directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.level.upper()))
        
        root_logger.handlers.clear()
        
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        
        file_handler = RotatingFileHandler(
            log_dir / self.config.logging.file_name,
            maxBytes=self.config.logging.max_size_mb * 1024 * 1024,
            backupCount=self.config.logging.backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        error_handler = RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("telegram").setLevel(logging.WARNING)
        logging.getLogger("aiosqlite").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        self.logger = logging.getLogger("main")
        return self.logger


class HealthMonitor:
    def __init__(self, bot: TradingBot, db: Database, logger: logging.Logger):
        self.bot = bot
        self.db = db
        self.logger = logger
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(60)
                
                db_health = await self.db.integrity_check()
                if db_health.get("status") != "ok" and db_health.get("status") != "skipped":
                    self.logger.error(f"Database integrity issue: {db_health}")
                    
                gc.collect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")


async def main() -> int:
    exit_code = 0
    shutdown = GracefulShutdown()
    bot: Optional[TradingBot] = None
    db: Optional[Database] = None
    health_monitor: Optional[HealthMonitor] = None
    audit_logger: Optional[AuditLogger] = None
    
    try:
        config.initialize()
        
        app_logger = ApplicationLogger(config)
        logger = app_logger.setup()
        
        issues = config.validate()
        for issue in issues:
            logger.warning(f"Config issue: {issue}")
            
        logger.info("=" * 60)
        logger.info("SOLANA TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Version: 1.0.0")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Network: {config.solana.network}")
        logger.info(f"RPC URL: {config.solana.rpc_url[:50]}...")
        logger.info(f"Paper Trading: {config.trading.paper_trading}")
        logger.info(f"Database: {config.database.path}")
        logger.info("=" * 60)
        
        db = Database(
            db_path=config.database.path,
            max_connections=config.database.max_connections,
            query_timeout=config.database.query_timeout
        )
        await db.initialize()
        shutdown.register_callback(db.close)
        logger.info("Database initialized")
        
        audit_logger = AuditLogger(db)
        await audit_logger.log_event(
            event_type="system",
            action="startup",
            details={"version": "1.0.0", "network": config.solana.network}
        )
        
        bot = get_bot()
        shutdown.register_callback(bot.stop)
        logger.info("Trading bot initialized")
        
        health_monitor = HealthMonitor(bot, db, logger)
        await health_monitor.start()
        shutdown.register_callback(health_monitor.stop)
        
        loop = asyncio.get_event_loop()
        
        def signal_handler(sig):
            logger.info(f"Received signal {sig.name}")
            asyncio.create_task(shutdown.trigger_shutdown())
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
            except NotImplementedError:
                signal.signal(sig, lambda s, f, sig=sig: signal_handler(sig))
                
        await bot.start()
        logger.info("Bot started, waiting for shutdown signal...")
        
        await shutdown.shutdown_event.wait()
        
        logger.info("Shutdown initiated...")
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.error(traceback.format_exc())
        exit_code = 1
    finally:
        try:
            if audit_logger and db:
                await audit_logger.log_event(
                    event_type="system",
                    action="shutdown",
                    details={"exit_code": exit_code}
                )
        except:
            pass
            
        await shutdown.trigger_shutdown()
        
        await asyncio.sleep(0.5)
        
        logging.info("Shutdown complete")
        
    return exit_code


def run() -> None:
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()