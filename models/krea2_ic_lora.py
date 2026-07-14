"""Improved non-stitched IC-LoRA for Krea 2."""

from models.krea2_reference import Krea2ReferencePipeline


class Krea2ICLoRAPipeline(Krea2ReferencePipeline):
    name = 'krea2_ic_lora'
    config_section = 'krea2_ic_lora'
