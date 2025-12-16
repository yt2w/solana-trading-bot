"""
Main entry point for the Solana Trading Bot.

SECURITY: This bot requires proper configuration of encryption secrets.
The bot will NOT start if ENCRYPTION_SECRET and ENCRYPTION_SALT are not set.
"""

import asyncio
import logging
import sys
from telegram.ext import Application, CommandHandler, CallbackQueryHandler

from .config import config
from .handlers import TradingBotHandlers
from .audit import init_audit_logger, AuditAction, AuditResult


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def post_init(application):
    """Called after application initialization."""
    logger.info("Bot initialized successfully!")


async def post_shutdown(application):
    """Called after application shutdown."""
    logger.info("Bot shut down.")
    # Log shutdown
    from .audit import get_audit_logger
    audit = get_audit_logger()
    audit.log(
        action=AuditAction.SYSTEM_SHUTDOWN,
        result=AuditResult.SUCCESS,
        details={"message": "Bot shutdown complete"}
    )


def main():
    """Main function to run the bot."""
    logger.info("=" * 60)
    logger.info("Solana Trading Bot - Starting")
    logger.info("=" * 60)
    
    # SECURITY: Validate ALL configuration including security-critical settings
    # This will exit immediately if encryption secrets are not configured
    logger.info("Validating configuration...")
    config.validate_all_and_exit_on_critical()
    
    # Initialize audit logger
    logger.info("Initializing audit logging...")
    audit = init_audit_logger(config.audit_log_path)
    
    # Log startup with security validation success
    audit.log(
        action=AuditAction.SYSTEM_STARTUP,
        result=AuditResult.SUCCESS,
        details={
            "message": "Bot starting with validated security configuration",
            "encryption_configured": True,
            "audit_log_path": config.audit_log_path
        }
    )
    
    logger.info("Starting Solana Trading Bot...")
    
    # Create application
    application = (
        Application.builder()
        .token(config.telegram_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    
    # Initialize handlers
    try:
        handlers = TradingBotHandlers()
    except Exception as e:
        logger.critical(f"Failed to initialize handlers: {e}")
        audit.log(
            action=AuditAction.SYSTEM_ERROR,
            result=AuditResult.FAILURE,
            details={"error": str(e), "phase": "handler_init"}
        )
        sys.exit(1)
    
    # Register all handlers
    for handler in handlers.get_handlers():
        application.add_handler(handler)
    
    # Add error handler
    async def error_handler(update, context):
        logger.error(f"Exception while handling an update: {context.error}")
        audit.log(
            action=AuditAction.SYSTEM_ERROR,
            result=AuditResult.ERROR,
            details={"error": str(context.error)[:200]}
        )
    
    application.add_error_handler(error_handler)
    
    # Run the bot
    logger.info("Bot is running. Press Ctrl+C to stop.")
    logger.info("=" * 60)
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
