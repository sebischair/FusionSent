# This module encapsulates the set of training arguments that can be passed to the FusionSent model to specifiy training.

from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass
from sentence_transformers import losses
import warnings

@dataclass
class TrainingArguments:
    """
    A dataclass containing all the arguments that can be passed to the FusionSent model to specifiy training.
    Pass these either at model initialisation (to be the same for all training runs), or specifically, when calling the training method.

    FusionSent trains two distinct sub-models, 'setfit' and 'label_embedding', whichs parameters are then fused.
    For customization purposes, the majority of training arguments can hence be given as a Tuple, in which the first and seccond components are destined to the "set-fit"- and "label-embedding"-submodel respectivley.
    If only a single value is provided, it will be used for both sub-models.

    After instantiation, each model's specific traning arguments are referenceable through custom properties of this class.
    Example:
        batch_sizes[0] is addressed to 'setfit', accessible as property 'TrainingArguments.setfit_batch_size'.
        batch_sizes[1] is addressed to 'label_embedding', accessible as property 'Trainingarguments.label_embedding_batch_size'.
    
    Attributes:
        batch_sizes (Optional[Union[int, Tuple[int, int]]]): Batch sizes for training. Single integer for both sub-models, or a tuple, to address each one individually. Default is (16, 1).
        num_epochs (Optional[Union[int, Tuple[int, int]]]): Number of epochs for training. Single integer for both sub-models, or a tuple, to address each one individually. Default is (1, 3).
        sampling_strategies (Optional[Union[str, Tuple[str, str]]]): Sampling strategies for training data. Single string for both sub-models, or a tuple, to address each one individually. Choose either "oversampling" (Default), "unique", or "undersampling", respectivley. See 'setfit.ContrastiveDataset' for more details.
        num_iterations (Optional[int]): Number of iterations for training. Always the same for both sub-models.
        distance_metrics (Optional[Union[Callable, Tuple[Callable, Callable]]]): Distance metrics for loss functions. Single 'Callable' for both sub-models, or a tuple, to address each one individually. Default is cosine distance for triplet loss.
        losses (Optional[Union[Callable, Tuple[Callable, Callable]]]): Loss functions for training. Single 'Callable' for both sub-models, or a tuple, to address each one individually Default is (CosineSimilarityLoss, ContrastiveLoss).
        merging_method (Optional[str]): Method for merging the parameters of both sub-modules after training. Choose either 'slerp' (default), or 'lerp'.
        margins (Optional[Union[float, Tuple[float, float]]]): Margin values for loss functions, to determine the threshold for considering examples as similar or dissimilar. Single float for both models, or a tuple, to address each one individually. Default is 0.25.
        warmup_proportions (Optional[Union[float, Tuple[float, float]]]): Proportion of the total training steps used for warming up the learning rates. Single float for both models, or a tuple, to address each one individually. Default is 0.1.
        samples_per_label (Optional[Union[int, Tuple[int, int]]]): Number of samples per label for training. A single integer for both models, or a tuple, to address each one individually. Default is 2.
        show_progress_bar (Optional[bool]): Whether to show progress bar during training. Default is True.
        use_setfit_body (Optional[bool]): Whether to train the 'setfit' submodel, and use its parameters in the merged FusionSent, or not. Use this when you only want to evaluate the 'label_embedding' sub-model. Default is True.
        json_path (Optional[str]): Path to save evaluation results as JSON.
    """

    batch_sizes: Optional[Union[int, Tuple[int, int]]] = (16, 1)
    num_epochs: Optional[Union[int, Tuple[int, int]]] = (1, 3)
    sampling_strategies: Optional[Union[str, Tuple[str, str]]] = "oversampling"
    num_iterations: Optional[int] = None
    distance_metrics: Optional[Union[Callable, Tuple[Callable, Callable]]] = losses.BatchHardTripletLossDistanceFunction.cosine_distance
    losses: Optional[Union[Callable, Tuple[Callable, Callable]]] = (losses.CosineSimilarityLoss, losses.ContrastiveLoss)
    merging_method: Optional[str] = 'slerp'
    margins: Optional[Union[float, Tuple[float, float]]] = 0.25
    warmup_proportions: Optional[Union[float, Tuple[float, float]]] = 0.1
    samples_per_label: Optional[Union[int, Tuple[int, int]]] = 2
    show_progress_bar: Optional[bool] = True
    use_setfit_body: Optional[bool] = True
    json_path: Optional[str] = None

    @property
    def setfit_batch_size(self) -> int:
        """
        Batch sizes for training the 'setfit' sub-model.
        """
        if isinstance(self.batch_sizes, int):
            return self.batch_sizes
        else:
            return self.batch_sizes[0]

    @property
    def label_embedding_batch_size(self) -> int:
        """
        Batch sizes for training the 'label_embedding' sub-model.
        """
        if isinstance(self.batch_sizes, int):
            return self.batch_sizes
        else:
            return self.batch_sizes[1]

    @property
    def setfit_num_epochs(self) -> int:
        """
        Number of epochs for training the 'setfit' sub-model.
        """
        if isinstance(self.num_epochs, int):
            return self.num_epochs
        else:
            return self.num_epochs[0]

    @property
    def label_embedding_num_epochs(self) -> int:
        """
        Number of epochs for training the 'label_embedding' sub-model.
        """
        if isinstance(self.num_epochs, int):
            return self.num_epochs
        else:
            return self.num_epochs[1]

    @property
    def setfit_sampling_strategy(self) -> str:
        """
        Sampling strategy for training data of the 'setfit' sub-model.
        Either "oversampling" (Default), "unique", or "undersampling".
        See 'setfit.ContrastiveDataset' for more details.
        """
        if isinstance(self.sampling_strategies, str):
            return self.sampling_strategies
        else:
            return self.sampling_strategies[0]

    @property
    def label_embedding_sampling_strategy(self) -> str:
        """
        Sampling strategy for training data of the 'label_embedding' sub-model.
        Either "oversampling" (Default), "unique", or "undersampling".
        See 'setfit.ContrastiveDataset' for more details.
        """
        if isinstance(self.sampling_strategies, str):
            return self.sampling_strategies
        else:
            return self.sampling_strategies[1]

    @property
    def setfit_distance_metric(self) -> Callable:
        """
        Distance metric for the loss function of the 'setfit' sub-model.
        """
        if isinstance(self.distance_metrics, Callable):
            return self.distance_metrics
        else:
            return self.distance_metrics[0]

    @property
    def label_embedding_distance_metric(self) -> Callable:
        """
        Distance metric for the loss function of the 'label_embedding' sub-model.
        """
        if isinstance(self.distance_metrics, Callable):
            return self.distance_metrics
        else:
            return self.distance_metrics[1]

    @property
    def setfit_loss(self) -> Callable:
        """
        Loss function for training the 'setfit' sub-model.
        """
        if isinstance(self.losses, Callable):
            return self.losses
        else:
            return self.losses[0]

    @property
    def label_embedding_loss(self) -> Callable:
        """
        Loss function for training the 'label_embedding' sub-model.
        """
        if isinstance(self.losses, Callable):
            return self.losses
        else:
            return self.losses[1]

    @property
    def setfit_margin(self) -> float:
        """
        Margin values for the loss function of the 'setfit' sub-model.
        This determines the threshold for considering examples as similar or dissimilar.
        """
        if isinstance(self.margins, float):
            return self.margins
        else:
            return self.margins[0]

    @property
    def label_embedding_margin(self) -> float:
        """
        Margin values for the loss function of the 'label_embedding' sub-model.
        This determines the threshold for considering examples as similar or dissimilar.
        """
        if isinstance(self.margins, float):
            return self.margins
        else:
            return self.margins[1]

    @property
    def setfit_warmup_proportion(self) -> float:
        """
        Proportion of the total training steps used for warming up the learning rate for training the 'setfit' sub-model.
        """
        if isinstance(self.warmup_proportions, float):
            return self.warmup_proportions
        else:
            return self.warmup_proportions[0]

    @property
    def label_embedding_warmup_proportion(self) -> float:
        """
        Proportion of the total training steps used for warming up the learning rate for training the 'label_embedding' sub-model.
        """
        if isinstance(self.warmup_proportions, float):
            return self.warmup_proportions
        else:
            return self.warmup_proportions[1]

    @property
    def setfit_samples_per_label(self) -> int:
        """
        Number of samples per label for training the 'setfit' submodel.
        """
        if isinstance(self.samples_per_label, int):
            return self.samples_per_label
        else:
            return self.samples_per_label[0]

    @property
    def label_embedding_samples_per_label(self) -> int:
        """
        Number of samples per label for training the 'label_embedding' submodel.
        """
        if isinstance(self.samples_per_label, int):
            return self.samples_per_label
        else:
            return self.samples_per_label[1]

    def _validate(self):
        """
        Validates the provided training arguments to ensure they are in the correct format and contain necessary values.
        Raises warnings for missing optional arguments and exceptions for missing non-optional ones.
        """
        # Optional warning for missing json_path
        if self.json_path is None:
            warnings.warn(
                f"`{self.__class__.__name__}.train` did not receive a `json_path`."
                f"Evaluation results will not be saved to file."
                f"Please provide a `json_path` to the `TrainingArguments` instance to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Validate required fields by accessing properties and catching errors
        required_properties = [
            ('setfit_batch_size', int), ('label_embedding_batch_size', int),
            ('setfit_num_epochs', int), ('label_embedding_num_epochs', int),
            ('setfit_sampling_strategy', str), ('label_embedding_sampling_strategy', str),
            ('setfit_distance_metric', Callable), ('label_embedding_distance_metric', Callable),
            ('setfit_loss', Callable), ('label_embedding_loss', Callable),
            ('setfit_margin', float), ('label_embedding_margin', float),
            ('setfit_warmup_proportion', float), ('label_embedding_warmup_proportion', float),
            ('setfit_samples_per_label', int), ('label_embedding_samples_per_label', int)
        ]
        for prop, expected_type in required_properties:
            try:
                value = getattr(self, prop)
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected type {expected_type} for {prop}, but got {type(value)}.")
            except Exception as e:
                raise ValueError(f"Invalid value for {prop}: {str(e)}")

        # Check for valid values in sampling_strategies
        valid_sampling_strategies = {"oversampling", "unique", "undersampling"}
        sampling_strategy_props = ['setfit_sampling_strategy', 'label_embedding_sampling_strategy']
        for strategy_prop in sampling_strategy_props:
            value = getattr(self, strategy_prop)
            if value not in valid_sampling_strategies:
                    raise ValueError(f"Invalid value '{value}' for '{prop}'. Must be one of {valid_sampling_strategies}.")