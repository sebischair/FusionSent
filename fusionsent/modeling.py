# This module contains the actual FusionSent model, with additional sub-models for different prediction strategies and classification heads.

from typing import  Callable, Dict,  List, Optional, Union
import warnings
from packaging.version import Version, parse
from dataclasses import dataclass, field
import copy
import numpy as np
import torch
from huggingface_hub.utils import validate_hf_hub_args
from huggingface_hub import PyTorchModelHubMixin
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import __version__ as sentence_transformers_version
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

class FusionModelBody:
    """
    This class encapsulates the dual encoder bodies of all variants ('setfit', 'label_embedding', 'fusion') for the FusionSent model.

    Attributes:
        setfit_model_body (SentenceTransformer): A copy of the SentenceTransformer model for setfit.
        label_embedding_model_body (SentenceTransformer): A copy of the SentenceTransformer model for label embedding.
        fusion_model_body (SentenceTransformer): A copy of the SentenceTransformer model for fusion.
    """

    def __init__(self, model: SentenceTransformer):
        self.setfit_model_body = copy.deepcopy(model)
        self.label_embedding_model_body = copy.deepcopy(model)
        self.fusion_model_body = copy.deepcopy(model)


class FusionModelHead:
    """
    This class to encapsulate the classification heads for all variants of encoder bodies ('setfit', 'label_embedding', 'fusion') of the FusionSent model.

    Attributes:
        setfit_model_head (Callable): A copy of the classification head for setfit.
        label_embedding_model_head (Callable): A copy of the classification head for label embedding.
        fusion_model_head (Callable): A copy of the classification head for fusion.
    """

    def __init__(self, model: Callable):
        self.setfit_model_head = copy.deepcopy(model)
        self.label_embedding_model_head = copy.deepcopy(model)
        self.fusion_model_head = copy.deepcopy(model)

@dataclass
class FusionSentModel(PyTorchModelHubMixin):
    """
    This data class for the FusionSent model includes model bodies and heads for different prediction strategies.

    The FusionSentModel is designed to encapsulate three separate sub-models, each with a pretrained language model at its core, and a linear classification head on top:
    - `setfit`: An encoder (body) intended to be trained contrastivley, with regular (item, item)-pairs (adapted from https://github.com/huggingface/setfit).
    - `label_embedding`: An encoder (body) intended to be trained with pairs of (class-descriptions, item)-pairs.
    - `fusion`: An encoder (body) that is the result of an (spherical) linear interpolation between the parameters of both the `setfit` and `label_embedding` sub-models.

    Each sub-model makes up a unique 'prediction strategy'. I.e., each sub-model (encoder + classification head) can be selected at runtime to be used.
    Only one sub-model can be selected at any given time. 

    Attributes:
        model_body (FusionModelBody): An instance of FusionModelBody containing the model bodies ('fusion', 'label_embedding', 'setfit').
        model_head (FusionModelHead): An instance of FusionModelHead containing the model heads ('fusion', 'label_embedding', 'setfit').
        multi_target_strategy (Optional[str]): The strategy for handling multi-target classification ('one-vs-rest', 'multi-output', or 'classifier-chain').
        prediction_strategy (Optional[str]): The current prediction strategy ('fusion', 'label_embedding', 'setfit').
        sentence_transformers_kwargs (Dict): Additional keyword arguments for SentenceTransformer implementation.
        transformers_config (Optional[Dict]): Configuration for the transformer implementation.
    """

    model_body: Optional[FusionModelBody] = None
    model_head: Optional[FusionModelHead] = None
    multi_target_strategy: Optional[str] = None
    prediction_strategy: Optional[str] = None
    sentence_transformers_kwargs: Dict = field(default_factory=dict, repr=False)
    transformers_config: Optional[Dict] = None

    def get_prediction_strategy(self)->str:
        """"
        Returns the prediction strategy for the model body. If not `None`, it can be either `fusion`, `label_embedding` or `setfit`.
        """
        return self.prediction_strategy

    def set_prediction_strategy(self, prediction_strategy: str)->None:
        """
        Sets the prediction strategy of the model body. If not `None`, it can be either `fusion`, `label_embedding` or `setfit`.
        Args:
            prediction_strategy (`str`):  A string representing the prediction strategy of the model. If not `None`, it can be either `fusion`, `label_embedding` or `setfit`.
        """
        self.prediction_strategy = prediction_strategy

    def encode(self, texts: List[str], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))->np.ndarray:
        """
        Convert input texts to embeddings using the SentenceTransformer dual encoder body.

        Args:
            texts (`List[str]`): A list of texts to encode.
        """
        if self.get_prediction_strategy() is None or self.get_prediction_strategy() == "fusion":
            # get fusion embeddings
            embeddings = self.get_fusion_embeddings(texts, device=device)
        elif self.get_prediction_strategy() == "setfit":
            # get SetFit embeddings
            embeddings = self.get_setfit_embeddings(texts, device=device)
        elif self.get_prediction_strategy() == "label_embedding":
            # get label embeddings
            embeddings = self.get_label_embeddings(texts, device=device)

        return embeddings

    def get_fusion_embeddings(self, texts: List[str], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))->np.ndarray:
        """
        Convert input texts to embeddings using the fusion model body.

        Args:
            texts (`List[str]`): A list of texts to encode.
        """
        # get embeddings from fusion body
        return self.model_body.fusion_model_body.encode(texts, device=device)

    def get_setfit_embeddings(self, texts: List[str], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))->np.ndarray:
        """
        Convert input texts to embeddings using the SetFit model body.

        Args:
            texts (`List[str]`): A list of texts to encode.
        """
        # get embeddings from SetFit body
        return self.model_body.setfit_model_body.encode(texts, device=device)

    def get_label_embeddings(self, texts: List[str], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))->np.ndarray:
        """
        Convert input texts to embeddings using the label embeddings model body.

        Args:
            texts (`List[str]`): A list of texts to encode.
        """
        # get embeddings from label embeddings body
        return self.model_body.label_embedding_model_body.encode(texts, device=device)

    def predict(self, texts: List[str], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))->np.ndarray:
        """
        Predict classes of input texts.

        Args:
            texts (`List[str]`): A list of texts for classification.
        """
        # encode texts as input features for classification head
        features = self.encode(texts, device=device)

        # classify texts
        if self.get_prediction_strategy() is None or self.get_prediction_strategy() == "fusion":
            # get fusion head embeddings
            predictions = self.model_head.fusion_model_head.predict(features)
        elif self.get_prediction_strategy() == "setfit":
            # get SetFit head predictions
            predictions = self.model_head.setfit_model_head.predict(features)
        elif self.get_prediction_strategy() == "label_embedding":
            # get label embedding head predictions
            predictions = self.model_head.label_embedding_model_head.predict(features)

        return predictions

    @classmethod
    @validate_hf_hub_args
    def _from_pretrained(
            cls,
            model_id: str,
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: Optional[bool] = None,
            proxies: Optional[Dict] = None,
            resume_download: Optional[bool] = None,
            local_files_only: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
            multi_target_strategy: Optional[str] = None,
            prediction_strategy: Optional[str] = None,
            device: Optional[Union[torch.device, str]] = None,
            trust_remote_code: bool = False,
            **model_kwargs,
    ) -> 'FusionSentModel':
        """
        Internal method to load a pretrained FusionSent model from the Hugging Face Hub.

        This method is called by the Hugging Face Hub framework and should not be modified by the user.
        It initializes the FusionSent model with pretrained components and configuration from the Hugging Face Hub.

        Args:
            model_id (str): The ID of the model on the Hugging Face Hub.
            cache_dir (Optional[str], optional): Directory to cache the model.
            token (Optional[Union[bool, str]], optional): Token for accessing the Hub.
            multi_target_strategy (Optional[str], optional): Strategy for multi-target classification ('one-vs-rest', 'multi-output', or 'classifier-chain').
            prediction_strategy (Optional[str], optional): The prediction strategy to use.
            device (Optional[Union[torch.device, str]], optional): The device to use for the model.
            trust_remote_code (bool, optional): Whether to trust custom code from the model repo.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            FusionSentModel: The loaded FusionSent model.
        """
        # Warn if any unused arguments are provided. -- Disabled this, because it will always be passed by parent class.
        # unused_args = [
        #     ('revision', revision),
        #     ('force_download', force_download),
        #     ('proxies', proxies),
        #     ('resume_download', resume_download),
        #     ('local_files_only', local_files_only)
        # ]
        # for arg_name, arg_value in unused_args:
        #     if arg_value is not None:
        #         warnings.warn(f"The '{arg_name}' argument is not used by 'FusionSentModel', and will have no effect.", UserWarning, stacklevel=2)

        #Setup additional arguments for sentence-transformer.
        sentence_transformers_kwargs = {
            "cache_folder": cache_dir,
            "use_auth_token": token,
            "device": device,
            "trust_remote_code": trust_remote_code,
        }
        if parse(sentence_transformers_version) >= Version("2.3.0"):
            sentence_transformers_kwargs = {
                "cache_folder": cache_dir,
                "token": token,
                "device": device,
                "trust_remote_code": trust_remote_code,
            }
        else:
            if trust_remote_code:
                raise ValueError(
                    "The `trust_remote_code` argument is only supported for `sentence-transformers` >= 2.3.0."
                )
            sentence_transformers_kwargs = {
                "cache_folder": cache_dir,
                "use_auth_token": token,
                "device": device,
            }

        #Load model components.
        word_embedding_model = models.Transformer(model_id)
        pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentence_transformer = SentenceTransformer(modules=[word_embedding_model, pooling_model], **sentence_transformers_kwargs)
        model_body = FusionModelBody(sentence_transformer)

        #Set device.
        if parse(sentence_transformers_version) >= Version("2.3.0"):
            device = sentence_transformer.device
        else:
            device = sentence_transformer._target_device

        #Configure classification-heads.
        head_params = model_kwargs.pop("head_params", {})
        clf = LogisticRegression(**head_params)
        if multi_target_strategy is not None:
            if multi_target_strategy == "one-vs-rest":
                multilabel_classifier = OneVsRestClassifier(clf)
            elif multi_target_strategy == "multi-output":
                multilabel_classifier = MultiOutputClassifier(clf)
            elif multi_target_strategy == "classifier-chain":
                multilabel_classifier = ClassifierChain(clf)
            else:
                raise ValueError(f"multi_target_strategy {multi_target_strategy} is not supported.")

            model_head = FusionModelHead(multilabel_classifier)
        else:
            model_head = FusionModelHead(clf)

        return cls(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy=multi_target_strategy,
            prediction_strategy=prediction_strategy,
            sentence_transformers_kwargs=sentence_transformers_kwargs,
            **model_kwargs,
        )