"""
Proof of History (PoH) Implementation Module

This module implements the Proof of History (PoH) consensus mechanism, which is a high-throughput
verification system for ordering events and transactions. PoH is a VDF (Verifiable Delay Function)
that produces a sequential hash chain, proving that a sequence of events occurred in a specific order
and were separated by a specific amount of time.

Key aspects of this implementation:
- Creates a sequential, cryptographically verifiable record of events
- Provides timestamp proofs that can be efficiently verified
- Enables high-throughput transaction processing by reducing consensus overhead
- Maintains a secure chain of hashed events that cannot be manipulated without detection

The ProofOfHistory class handles the creation and verification of this sequential hash chain,
generating proofs that can be independently verified by any network participant.
"""

import hashlib
import time
import json
from typing import Any, Optional


class ProofOfHistory:
    """
    Implements a Proof of History consensus mechanism for blockchain systems.
    
    The Proof of History is a sequence of computation that creates a verifiable delay 
    function and generates a unique, verifiable timestamp for each transaction or event.
    
    Attributes:
        sequence_number (int): The current position in the Proof of History sequence.
        last_hash (str): The hash of the last event in the PoH chain.
        last_timestamp (float): The timestamp of the last hash generation.
    """
    
    def __init__(self):
        """
        Initialize a new Proof of History chain with a genesis hash.
        """
        self.sequence_number = 0
        self.last_hash = hashlib.sha256("genesis".encode()).hexdigest()
        self.last_timestamp = time.time()
    
    def tick(self, data: Optional[Any] = None) -> str:
        """
        Generate a new proof of history tick, optionally incorporating provided data.
        
        This method creates a new hash based on the last hash, sequence number,
        and optional data. It increments the sequence number and updates the last hash
        and timestamp.
        
        Args:
            data: Optional data to incorporate into the hash calculation.
            
        Returns:
            str: The new hash generated in this tick.
        """
        self.sequence_number += 1
        current_time = time.time()
        
        # Prepare the input for hashing
        input_data = {
            "last_hash": self.last_hash,
            "sequence": self.sequence_number,
            "timestamp": current_time,
            "data": data
        }
        
        # Generate new hash
        new_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        
        # Update state
        self.last_hash = new_hash
        self.last_timestamp = current_time
        
        return new_hash
    
    def verify_tick(self, prev_hash: str, seq_num: int, timestamp: float, 
                   data: Optional[Any], resulting_hash: str) -> bool:
        """
        Verify that a given PoH tick is valid.
        
        This method checks if the provided hash is the result of correctly hashing
        the provided inputs according to the PoH rules.
        
        Args:
            prev_hash: The previous hash in the sequence.
            seq_num: The sequence number of this tick.
            timestamp: The timestamp when this tick was created.
            data: Any data incorporated into this tick.
            resulting_hash: The hash that was generated for this tick.
            
        Returns:
            bool: True if the hash verification succeeds, False otherwise.
        """
        # Recreate the input data with the provided parameters
        input_data = {
            "last_hash": prev_hash,
            "sequence": seq_num,
            "timestamp": timestamp,
            "data": data
        }
        
        # Compute the hash
        computed_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        
        # Verify it matches the provided hash
        return computed_hash == resulting_hash
    
    def get_state(self) -> dict:
        """
        Get the current state of the PoH chain.
        
        Returns:
            dict: A dictionary containing the current sequence number, last hash,
                  and last timestamp.
        """
        return {
            "sequence_number": self.sequence_number,
            "last_hash": self.last_hash,
            "last_timestamp": self.last_timestamp
        }

