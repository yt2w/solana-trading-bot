
import asyncio
import time
import random
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Union, Any
import logging
import base64

import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment, Confirmed
from solana.rpc.types import TxOpts
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.hash import Hash
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction
from solders.system_program import TransferParams, transfer
from solders.address_lookup_table_account import AddressLookupTableAccount

logger = logging.getLogger(__name__)

class CommitmentLevel(str, Enum):
    PROCESSED = "processed"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"

class TransactionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    FAILED = "failed"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"

class JitoRegion(str, Enum):
    MAINNET = "mainnet"
    AMSTERDAM = "amsterdam"
    FRANKFURT = "frankfurt"
    NEW_YORK = "ny"
    TOKYO = "tokyo"

JITO_ENDPOINTS = {
    JitoRegion.MAINNET: "https://mainnet.block-engine.jito.wtf",
    JitoRegion.AMSTERDAM: "https://amsterdam.mainnet.block-engine.jito.wtf",
    JitoRegion.FRANKFURT: "https://frankfurt.mainnet.block-engine.jito.wtf",
    JitoRegion.NEW_YORK: "https://ny.mainnet.block-engine.jito.wtf",
    JitoRegion.TOKYO: "https://tokyo.mainnet.block-engine.jito.wtf",
}

JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4bVmkdzGHb67ETqsnjhJZeK",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]

COMPUTE_BUDGET_PROGRAM_ID = Pubkey.from_string("ComputeBudget111111111111111111111111111111")

DEFAULT_COMPUTE_UNITS = 200_000
DEFAULT_COMPUTE_UNIT_PRICE = 1_000
DEFAULT_CONFIRMATION_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.5

class TransactionError(Exception):
    def __init__(self, message: str, logs: Optional[List[str]] = None):
        super().__init__(message)
        self.logs = logs or []

class TransactionBuildError(TransactionError):
    pass

class TransactionSignError(TransactionError):
    pass

class TransactionSendError(TransactionError):
    def __init__(self, message: str, error_code: Optional[int] = None, logs: Optional[List[str]] = None):
        super().__init__(message, logs)
        self.error_code = error_code

class TransactionConfirmError(TransactionError):
    pass

class TransactionExpiredError(TransactionError):
    pass

class TransactionSimulationError(TransactionError):
    def __init__(self, message: str, units_consumed: Optional[int] = None, logs: Optional[List[str]] = None):
        super().__init__(message, logs)
        self.units_consumed = units_consumed

class InsufficientFundsError(TransactionError):
    pass

class JitoError(TransactionError):
    pass

class BlockhashNotFoundError(TransactionError):
    pass

SOLANA_ERROR_CODES = {
    0: "Generic error",
    1: "Insufficient funds",
    2: "Invalid account data",
    3: "Account data too small",
    4: "Account not rent exempt",
    5: "Invalid instruction data",
    6: "Invalid account owner",
    7: "Arithmetic overflow",
    8: "Immutable account modified",
}

def parse_solana_error(error_data: Any) -> Tuple[str, Optional[int], List[str]]:
    logs = []
    error_code = None
    message = "Unknown error"
    
    if isinstance(error_data, dict):
        if "logs" in error_data:
            logs = error_data["logs"]
        
        if "err" in error_data:
            err = error_data["err"]
            if isinstance(err, dict):
                if "InstructionError" in err:
                    idx, inner_err = err["InstructionError"]
                    if isinstance(inner_err, dict):
                        error_type = list(inner_err.keys())[0]
                        message = f"Instruction {idx} failed: {error_type}"
                        if "Custom" in inner_err:
                            error_code = inner_err["Custom"]
                            message = f"Instruction {idx} failed with custom error {error_code}"
                    else:
                        message = f"Instruction {idx} failed: {inner_err}"
                else:
                    message = str(err)
            else:
                message = str(err)
        elif "message" in error_data:
            message = error_data["message"]
    elif isinstance(error_data, str):
        message = error_data
    
    return message, error_code, logs

def create_set_compute_unit_limit_instruction(units: int) -> Instruction:
    data = bytes([0x02]) + struct.pack("<I", units)
    return Instruction(
        program_id=COMPUTE_BUDGET_PROGRAM_ID,
        accounts=[],
        data=data
    )

def create_set_compute_unit_price_instruction(micro_lamports: int) -> Instruction:
    data = bytes([0x03]) + struct.pack("<Q", micro_lamports)
    return Instruction(
        program_id=COMPUTE_BUDGET_PROGRAM_ID,
        accounts=[],
        data=data
    )

def create_request_heap_frame_instruction(bytes_size: int) -> Instruction:
    data = bytes([0x01]) + struct.pack("<I", bytes_size)
    return Instruction(
        program_id=COMPUTE_BUDGET_PROGRAM_ID,
        accounts=[],
        data=data
    )

@dataclass
class CachedBlockhash:
    blockhash: Hash
    last_valid_block_height: int
    fetched_at: float
    
    def is_expired(self, max_age_seconds: float = 30) -> bool:
        return time.time() - self.fetched_at > max_age_seconds

class BlockhashCache:
    
    def __init__(self, max_age_seconds: float = 30):
        self.max_age = max_age_seconds
        self._cache: Optional[CachedBlockhash] = None
        self._lock = asyncio.Lock()
    
    async def get_blockhash(
        self,
        client: AsyncClient,
        force_refresh: bool = False
    ) -> Tuple[Hash, int]:
        async with self._lock:
            if force_refresh or self._cache is None or self._cache.is_expired(self.max_age):
                response = await client.get_latest_blockhash(commitment=Confirmed)
                
                if not response.value:
                    raise BlockhashNotFoundError("Failed to get recent blockhash")
                
                self._cache = CachedBlockhash(
                    blockhash=response.value.blockhash,
                    last_valid_block_height=response.value.last_valid_block_height,
                    fetched_at=time.time()
                )
                logger.debug(f"Fetched new blockhash: {self._cache.blockhash}")
            
            return self._cache.blockhash, self._cache.last_valid_block_height
    
    def invalidate(self):
        self._cache = None

_blockhash_cache = BlockhashCache()

async def get_recent_blockhash(
    client: AsyncClient,
    force_refresh: bool = False
) -> Tuple[Hash, int]:
    return await _blockhash_cache.get_blockhash(client, force_refresh)

@dataclass
class TransactionConfig:
    compute_units: Optional[int] = None
    compute_unit_price: Optional[int] = None
    use_versioned: bool = True
    skip_preflight: bool = False
    preflight_commitment: CommitmentLevel = CommitmentLevel.CONFIRMED
    max_retries: int = DEFAULT_MAX_RETRIES

class TransactionBuilder:
    
    def __init__(
        self,
        fee_payer: Pubkey,
        config: Optional[TransactionConfig] = None
    ):
        self.fee_payer = fee_payer
        self.config = config or TransactionConfig()
        self.instructions: List[Instruction] = []
        self.signers: List[Keypair] = []
        self.address_lookup_tables: List[AddressLookupTableAccount] = []
        self._blockhash: Optional[Hash] = None
        self._last_valid_block_height: Optional[int] = None
    
    def add_instruction(self, instruction: Instruction) -> "TransactionBuilder":
        self.instructions.append(instruction)
        return self
    
    def add_instructions(self, instructions: List[Instruction]) -> "TransactionBuilder":
        self.instructions.extend(instructions)
        return self
    
    def add_signer(self, signer: Keypair) -> "TransactionBuilder":
        self.signers.append(signer)
        return self
    
    def add_signers(self, signers: List[Keypair]) -> "TransactionBuilder":
        self.signers.extend(signers)
        return self
    
    def add_address_lookup_table(
        self,
        table: AddressLookupTableAccount
    ) -> "TransactionBuilder":
        self.address_lookup_tables.append(table)
        return self
    
    def set_compute_units(self, units: int) -> "TransactionBuilder":
        self.config.compute_units = units
        return self
    
    def set_priority_fee(self, micro_lamports: int) -> "TransactionBuilder":
        self.config.compute_unit_price = micro_lamports
        return self
    
    def set_blockhash(self, blockhash: Hash, last_valid_block_height: int) -> "TransactionBuilder":
        self._blockhash = blockhash
        self._last_valid_block_height = last_valid_block_height
        return self
    
    async def fetch_blockhash(self, client: AsyncClient) -> "TransactionBuilder":
        self._blockhash, self._last_valid_block_height = await get_recent_blockhash(client)
        return self
    
    def _build_instructions(self) -> List[Instruction]:
        instructions = []
        
        if self.config.compute_units is not None:
            instructions.append(
                create_set_compute_unit_limit_instruction(self.config.compute_units)
            )
        
        if self.config.compute_unit_price is not None:
            instructions.append(
                create_set_compute_unit_price_instruction(self.config.compute_unit_price)
            )
        
        instructions.extend(self.instructions)
        return instructions
    
    def build_legacy(self) -> Transaction:
        if not self._blockhash:
            raise TransactionBuildError("Blockhash not set. Call fetch_blockhash() first.")
        
        if not self.instructions:
            raise TransactionBuildError("No instructions added to transaction.")
        
        instructions = self._build_instructions()
        
        tx = Transaction(
            fee_payer=self.fee_payer,
            recent_blockhash=self._blockhash,
            instructions=instructions
        )
        
        return tx
    
    def build_versioned(self) -> VersionedTransaction:
        if not self._blockhash:
            raise TransactionBuildError("Blockhash not set. Call fetch_blockhash() first.")
        
        if not self.instructions:
            raise TransactionBuildError("No instructions added to transaction.")
        
        instructions = self._build_instructions()
        
        if self.address_lookup_tables:
            message = MessageV0.try_compile(
                payer=self.fee_payer,
                instructions=instructions,
                address_lookup_table_accounts=self.address_lookup_tables,
                recent_blockhash=self._blockhash
            )
        else:
            message = MessageV0.try_compile(
                payer=self.fee_payer,
                instructions=instructions,
                address_lookup_table_accounts=[],
                recent_blockhash=self._blockhash
            )
        
        tx = VersionedTransaction(message, [])
        return tx
    
    async def build(self, client: Optional[AsyncClient] = None) -> Union[Transaction, VersionedTransaction]:
        if not self._blockhash and client:
            await self.fetch_blockhash(client)
        
        if self.config.use_versioned:
            return self.build_versioned()
        else:
            return self.build_legacy()
    
    def clear(self) -> "TransactionBuilder":
        self.instructions.clear()
        self.signers.clear()
        self.address_lookup_tables.clear()
        self._blockhash = None
        self._last_valid_block_height = None
        return self

def sign_transaction(tx: Transaction, keypair: Keypair) -> Transaction:
    try:
        tx.sign(keypair)
        return tx
    except Exception as e:
        raise TransactionSignError(f"Failed to sign transaction: {e}")

def sign_transaction_multi(tx: Transaction, keypairs: List[Keypair]) -> Transaction:
    try:
        tx.sign(*keypairs)
        return tx
    except Exception as e:
        raise TransactionSignError(f"Failed to sign transaction: {e}")

def sign_versioned_transaction(
    tx: VersionedTransaction,
    keypair: Keypair
) -> VersionedTransaction:
    try:
        return VersionedTransaction(tx.message, [keypair])
    except Exception as e:
        raise TransactionSignError(f"Failed to sign versioned transaction: {e}")

def sign_versioned_transaction_multi(
    tx: VersionedTransaction,
    keypairs: List[Keypair]
) -> VersionedTransaction:
    try:
        return VersionedTransaction(tx.message, keypairs)
    except Exception as e:
        raise TransactionSignError(f"Failed to sign versioned transaction: {e}")

async def send_transaction(
    client: AsyncClient,
    tx: Union[Transaction, VersionedTransaction],
    skip_preflight: bool = False,
    preflight_commitment: CommitmentLevel = CommitmentLevel.CONFIRMED,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY
) -> Signature:
    opts = TxOpts(
        skip_preflight=skip_preflight,
        preflight_commitment=Commitment(preflight_commitment.value),
        max_retries=0
    )
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            if isinstance(tx, VersionedTransaction):
                response = await client.send_transaction(tx, opts=opts)
            else:
                response = await client.send_transaction(tx, opts=opts)
            
            if hasattr(response, 'value') and response.value:
                signature = response.value
                logger.info(f"Transaction sent: {signature}")
                return signature
            else:
                raise TransactionSendError("Empty response from send_transaction")
                
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            retryable = any(x in error_str for x in [
                "blockhash not found",
                "rate limit",
                "timeout",
                "connection",
                "429",
                "503",
                "504"
            ])
            
            if not retryable or attempt == max_retries:
                break
            
            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
            logger.warning(f"Transaction send failed (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {e}")
            await asyncio.sleep(wait_time)
    
    message, error_code, logs = parse_solana_error(str(last_error))
    raise TransactionSendError(message, error_code, logs)

async def send_raw_transaction(
    client: AsyncClient,
    raw_tx: bytes,
    skip_preflight: bool = False,
    preflight_commitment: CommitmentLevel = CommitmentLevel.CONFIRMED
) -> Signature:
    opts = TxOpts(
        skip_preflight=skip_preflight,
        preflight_commitment=Commitment(preflight_commitment.value)
    )
    
    response = await client.send_raw_transaction(raw_tx, opts=opts)
    
    if hasattr(response, 'value') and response.value:
        return response.value
    
    raise TransactionSendError("Failed to send raw transaction")

async def confirm_transaction(
    client: AsyncClient,
    signature: Signature,
    commitment: CommitmentLevel = CommitmentLevel.CONFIRMED,
    timeout: float = DEFAULT_CONFIRMATION_TIMEOUT,
    poll_interval: float = 0.5
) -> TransactionStatus:
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            status = await get_transaction_status(client, signature, commitment)
            
            if status == TransactionStatus.CONFIRMED:
                logger.info(f"Transaction confirmed: {signature}")
                return status
            elif status == TransactionStatus.FINALIZED:
                logger.info(f"Transaction finalized: {signature}")
                return status
            elif status == TransactionStatus.FAILED:
                raise TransactionConfirmError(f"Transaction failed: {signature}")
            elif status == TransactionStatus.EXPIRED:
                raise TransactionExpiredError(f"Transaction expired: {signature}")
            
            await asyncio.sleep(poll_interval)
            
        except TransactionConfirmError:
            raise
        except TransactionExpiredError:
            raise
        except Exception as e:
            logger.warning(f"Error checking transaction status: {e}")
            await asyncio.sleep(poll_interval)
    
    raise TransactionConfirmError(f"Transaction confirmation timeout after {timeout}s: {signature}")

async def get_transaction_status(
    client: AsyncClient,
    signature: Signature,
    commitment: CommitmentLevel = CommitmentLevel.CONFIRMED
) -> TransactionStatus:
    try:
        response = await client.get_signature_statuses([signature])
        
        if not response.value or not response.value[0]:
            return TransactionStatus.NOT_FOUND
        
        status = response.value[0]
        
        if status.err:
            return TransactionStatus.FAILED
        
        if status.confirmation_status:
            status_str = str(status.confirmation_status).lower()
            if "finalized" in status_str:
                return TransactionStatus.FINALIZED
            elif "confirmed" in status_str:
                return TransactionStatus.CONFIRMED
            elif "processed" in status_str:
                return TransactionStatus.PENDING
        
        return TransactionStatus.PENDING
        
    except Exception as e:
        logger.error(f"Error getting transaction status: {e}")
        return TransactionStatus.NOT_FOUND

async def send_and_confirm_transaction(
    client: AsyncClient,
    tx: Union[Transaction, VersionedTransaction],
    skip_preflight: bool = False,
    commitment: CommitmentLevel = CommitmentLevel.CONFIRMED,
    timeout: float = DEFAULT_CONFIRMATION_TIMEOUT
) -> Tuple[Signature, TransactionStatus]:
    signature = await send_transaction(client, tx, skip_preflight=skip_preflight)
    status = await confirm_transaction(client, signature, commitment, timeout)
    return signature, status

@dataclass
class SimulationResult:
    success: bool
    logs: List[str]
    units_consumed: Optional[int]
    error: Optional[str]
    accounts: Optional[List[Any]]

async def simulate_transaction(
    client: AsyncClient,
    tx: Union[Transaction, VersionedTransaction],
    commitment: CommitmentLevel = CommitmentLevel.CONFIRMED,
    sig_verify: bool = False,
    replace_recent_blockhash: bool = True
) -> SimulationResult:
    try:
        response = await client.simulate_transaction(
            tx,
            commitment=Commitment(commitment.value),
            sig_verify=sig_verify,
            replace_recent_blockhash=replace_recent_blockhash
        )
        
        if not response.value:
            return SimulationResult(
                success=False,
                logs=[],
                units_consumed=None,
                error="Empty simulation response",
                accounts=None
            )
        
        result = response.value
        
        return SimulationResult(
            success=result.err is None,
            logs=result.logs or [],
            units_consumed=result.units_consumed,
            error=str(result.err) if result.err else None,
            accounts=result.accounts
        )
        
    except Exception as e:
        return SimulationResult(
            success=False,
            logs=[],
            units_consumed=None,
            error=str(e),
            accounts=None
        )

async def estimate_compute_units(
    client: AsyncClient,
    tx: Union[Transaction, VersionedTransaction],
    buffer_percent: float = 20
) -> int:
    result = await simulate_transaction(client, tx)
    
    if not result.success:
        logger.warning(f"Simulation failed: {result.error}")
        return DEFAULT_COMPUTE_UNITS
    
    if result.units_consumed:
        buffered = int(result.units_consumed * (1 + buffer_percent / 100))
        return min(buffered, 1_400_000)
    
    return DEFAULT_COMPUTE_UNITS

async def estimate_priority_fee(
    client: AsyncClient,
    account_keys: Optional[List[Pubkey]] = None,
    percentile: int = 75
) -> int:
    try:
        if account_keys:
            response = await client.get_recent_prioritization_fees(account_keys)
        else:
            response = await client.get_recent_prioritization_fees([])
        
        if not response.value:
            return DEFAULT_COMPUTE_UNIT_PRICE
        
        fees = [f.prioritization_fee for f in response.value if f.prioritization_fee > 0]
        
        if not fees:
            return DEFAULT_COMPUTE_UNIT_PRICE
        
        fees.sort()
        idx = min(len(fees) - 1, int(len(fees) * percentile / 100))
        
        return max(fees[idx], DEFAULT_COMPUTE_UNIT_PRICE)
        
    except Exception as e:
        logger.warning(f"Error estimating priority fee: {e}")
        return DEFAULT_COMPUTE_UNIT_PRICE

@dataclass
class JitoBundleResult:
    bundle_id: str
    status: str
    landed_slot: Optional[int] = None
    error: Optional[str] = None

class JitoClient:
    
    def __init__(
        self,
        region: JitoRegion = JitoRegion.MAINNET,
        api_key: Optional[str] = None
    ):
        self.region = region
        self.api_key = api_key
        self.endpoint = JITO_ENDPOINTS[region]
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    def get_random_tip_account(self) -> Pubkey:
        return Pubkey.from_string(random.choice(JITO_TIP_ACCOUNTS))
    
    def create_tip_instruction(
        self,
        payer: Pubkey,
        tip_lamports: int,
        tip_account: Optional[Pubkey] = None
    ) -> Instruction:
        if tip_account is None:
            tip_account = self.get_random_tip_account()
        
        return transfer(
            TransferParams(
                from_pubkey=payer,
                to_pubkey=tip_account,
                lamports=tip_lamports
            )
        )
    
    async def send_bundle(
        self,
        transactions: List[Union[Transaction, VersionedTransaction]],
        timeout: float = 30
    ) -> JitoBundleResult:
        session = await self._get_session()
        
        serialized = []
        for tx in transactions:
            if isinstance(tx, VersionedTransaction):
                raw = bytes(tx)
            else:
                raw = tx.serialize()
            serialized.append(base64.b64encode(raw).decode())
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [serialized]
        }
        
        try:
            async with session.post(
                f"{self.endpoint}/api/v1/bundles",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                data = await response.json()
                
                if "error" in data:
                    error = data["error"]
                    raise JitoError(f"Bundle submission failed: {error}")
                
                bundle_id = data.get("result", "")
                
                return JitoBundleResult(
                    bundle_id=bundle_id,
                    status="submitted"
                )
                
        except aiohttp.ClientError as e:
            raise JitoError(f"Network error sending bundle: {e}")
    
    async def get_bundle_status(
        self,
        bundle_id: str,
        timeout: float = 10
    ) -> JitoBundleResult:
        session = await self._get_session()
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBundleStatuses",
            "params": [[bundle_id]]
        }
        
        try:
            async with session.post(
                f"{self.endpoint}/api/v1/bundles",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                data = await response.json()
                
                if "error" in data:
                    return JitoBundleResult(
                        bundle_id=bundle_id,
                        status="unknown",
                        error=str(data["error"])
                    )
                
                results = data.get("result", {}).get("value", [])
                
                if not results:
                    return JitoBundleResult(
                        bundle_id=bundle_id,
                        status="pending"
                    )
                
                status_data = results[0]
                
                return JitoBundleResult(
                    bundle_id=bundle_id,
                    status=status_data.get("confirmation_status", "pending"),
                    landed_slot=status_data.get("slot")
                )
                
        except aiohttp.ClientError as e:
            return JitoBundleResult(
                bundle_id=bundle_id,
                status="unknown",
                error=str(e)
            )
    
    async def send_and_confirm_bundle(
        self,
        transactions: List[Union[Transaction, VersionedTransaction]],
        timeout: float = 60,
        poll_interval: float = 1.0
    ) -> JitoBundleResult:
        result = await self.send_bundle(transactions)
        
        if result.error:
            return result
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_bundle_status(result.bundle_id)
            
            if status.status in ["confirmed", "finalized", "landed"]:
                return status
            elif status.error:
                return status
            
            await asyncio.sleep(poll_interval)
        
        return JitoBundleResult(
            bundle_id=result.bundle_id,
            status="timeout",
            error=f"Bundle confirmation timeout after {timeout}s"
        )

async def send_with_jito(
    tx: Union[Transaction, VersionedTransaction],
    tip_lamports: int,
    payer: Keypair,
    region: JitoRegion = JitoRegion.MAINNET,
    confirm: bool = True
) -> Union[JitoBundleResult, Signature]:
    async with JitoClient(region=region) as jito:
        if confirm:
            return await jito.send_and_confirm_bundle([tx])
        else:
            return await jito.send_bundle([tx])

async def build_and_send_transaction(
    client: AsyncClient,
    instructions: List[Instruction],
    payer: Keypair,
    signers: Optional[List[Keypair]] = None,
    compute_units: Optional[int] = None,
    priority_fee: Optional[int] = None,
    use_versioned: bool = True,
    skip_preflight: bool = False,
    confirm: bool = True,
    commitment: CommitmentLevel = CommitmentLevel.CONFIRMED
) -> Tuple[Signature, Optional[TransactionStatus]]:
    config = TransactionConfig(
        compute_units=compute_units,
        compute_unit_price=priority_fee,
        use_versioned=use_versioned,
        skip_preflight=skip_preflight
    )
    
    builder = TransactionBuilder(payer.pubkey(), config)
    builder.add_instructions(instructions)
    
    if signers:
        builder.add_signers(signers)
    
    await builder.fetch_blockhash(client)
    tx = await builder.build()
    
    all_signers = [payer] + (signers or [])
    
    if use_versioned:
        tx = sign_versioned_transaction_multi(tx, all_signers)
    else:
        tx = sign_transaction_multi(tx, all_signers)
    
    if confirm:
        signature, status = await send_and_confirm_transaction(
            client, tx, skip_preflight, commitment
        )
        return signature, status
    else:
        signature = await send_transaction(client, tx, skip_preflight)
        return signature, None

class TransactionManager:
    
    def __init__(
        self,
        client: AsyncClient,
        payer: Keypair,
        default_config: Optional[TransactionConfig] = None
    ):
        self.client = client
        self.payer = payer
        self.default_config = default_config or TransactionConfig()
        self.blockhash_cache = BlockhashCache()
    
    async def get_blockhash(self, force_refresh: bool = False) -> Tuple[Hash, int]:
        return await self.blockhash_cache.get_blockhash(self.client, force_refresh)
    
    def create_builder(self, config: Optional[TransactionConfig] = None) -> TransactionBuilder:
        return TransactionBuilder(
            self.payer.pubkey(),
            config or self.default_config
        )
    
    async def send(
        self,
        tx: Union[Transaction, VersionedTransaction],
        skip_preflight: bool = False
    ) -> Signature:
        return await send_transaction(
            self.client,
            tx,
            skip_preflight=skip_preflight or self.default_config.skip_preflight
        )
    
    async def confirm(
        self,
        signature: Signature,
        commitment: CommitmentLevel = CommitmentLevel.CONFIRMED,
        timeout: float = DEFAULT_CONFIRMATION_TIMEOUT
    ) -> TransactionStatus:
        return await confirm_transaction(
            self.client,
            signature,
            commitment,
            timeout
        )
    
    async def send_and_confirm(
        self,
        tx: Union[Transaction, VersionedTransaction],
        skip_preflight: bool = False,
        commitment: CommitmentLevel = CommitmentLevel.CONFIRMED,
        timeout: float = DEFAULT_CONFIRMATION_TIMEOUT
    ) -> Tuple[Signature, TransactionStatus]:
        return await send_and_confirm_transaction(
            self.client,
            tx,
            skip_preflight=skip_preflight or self.default_config.skip_preflight,
            commitment=commitment,
            timeout=timeout
        )
    
    async def simulate(
        self,
        tx: Union[Transaction, VersionedTransaction]
    ) -> SimulationResult:
        return await simulate_transaction(self.client, tx)
    
    async def estimate_fees(
        self,
        instructions: List[Instruction],
        account_keys: Optional[List[Pubkey]] = None
    ) -> Tuple[int, int]:
        builder = self.create_builder()
        builder.add_instructions(instructions)
        await builder.fetch_blockhash(self.client)
        tx = await builder.build()
        
        if isinstance(tx, VersionedTransaction):
            tx = sign_versioned_transaction(tx, self.payer)
        else:
            tx = sign_transaction(tx, self.payer)
        
        compute_units = await estimate_compute_units(self.client, tx)
        priority_fee = await estimate_priority_fee(self.client, account_keys)
        
        return compute_units, priority_fee

__all__ = [
    "CommitmentLevel",
    "TransactionStatus",
    "JitoRegion",
    
    "TransactionError",
    "TransactionBuildError",
    "TransactionSignError",
    "TransactionSendError",
    "TransactionConfirmError",
    "TransactionExpiredError",
    "TransactionSimulationError",
    "InsufficientFundsError",
    "JitoError",
    "BlockhashNotFoundError",
    
    "TransactionConfig",
    "SimulationResult",
    "JitoBundleResult",
    "CachedBlockhash",
    
    "TransactionBuilder",
    
    "sign_transaction",
    "sign_transaction_multi",
    "sign_versioned_transaction",
    "sign_versioned_transaction_multi",
    
    "send_transaction",
    "send_raw_transaction",
    "send_with_jito",
    
    "confirm_transaction",
    "get_transaction_status",
    "send_and_confirm_transaction",
    
    "simulate_transaction",
    "estimate_compute_units",
    "estimate_priority_fee",
    
    "get_recent_blockhash",
    "BlockhashCache",
    
    "create_set_compute_unit_limit_instruction",
    "create_set_compute_unit_price_instruction",
    "create_request_heap_frame_instruction",
    
    "JitoClient",
    "JITO_ENDPOINTS",
    "JITO_TIP_ACCOUNTS",
    
    "build_and_send_transaction",
    "TransactionManager",
    
    "parse_solana_error",
    "SOLANA_ERROR_CODES",
]
