"""OminiControl2 for Krea 2 with compact, independent reference tokens."""

from models.krea2_ominicontrol import Krea2OminiControlPipeline


class Krea2OminiControl2Pipeline(Krea2OminiControlPipeline):
    name = 'krea2_ominicontrol2'

    def __init__(self, config):
        super().__init__(config)
        control = config.get('ominicontrol', {})
        self.independent_condition = bool(control.get('independent_condition', True))
        self.condition_token_stride = int(control.get('condition_token_stride', 2))
        self.reference_position_scale = float(
            control.get('reference_position_scale', self.condition_token_stride)
        )
        if self.condition_token_stride < 1:
            raise ValueError('condition_token_stride must be >= 1')

    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        metadata['control_family'] = 'ominicontrol_v2'
        return metadata
