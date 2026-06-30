# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

from src.primitives.types import Primitive


class SaveReceipt(Primitive):
    @property
    def name(self):
        return "save_receipt"

    def guidelines(self):
        return "Save the receipt to the database"

    def execute(self, receipt_id: str, amount_paid: float):
        print(f"Saving receipt {receipt_id} with amount {amount_paid} to the database")
        return f"Receipt {receipt_id} saved with amount {amount_paid}", True
