"""Load only the public profile, with embedded contract and manifest checks."""
import torch
from .inference import InferenceService, Informer
from .public_preprocessing import PublicPreprocessingArtifacts


class PublicInferenceService(InferenceService):
    def load(self):
        self.model=None
        self.preprocessing=None
        if self.model_path.name != 'informer_checkpoint.pth':
            raise ValueError('Public checkpoint filename is invalid.')
        preprocessing=PublicPreprocessingArtifacts(self.model_path.parent)
        checkpoint=torch.load(self.model_path,map_location=self.device,weights_only=True)
        if checkpoint.get('contract') != preprocessing.metadata:
            raise ValueError('Checkpoint and public preprocessing contracts do not match.')
        config=preprocessing.metadata['model_config']
        if config.get('input_dim') != preprocessing.input_dim or config.get('seq_len') != preprocessing.input_window:
            raise ValueError('Public model architecture does not match its feature contract.')
        model=Informer(**config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        preprocessing.verified_energy_bundle=True
        self.preprocessing,self.model=preprocessing,model
