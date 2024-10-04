#This module contains the Trainer class, responsible for managing the training and evaluation process for FusionSent. 

from typing import Any,  Dict, Iterable, List, Optional, Tuple, Union
import warnings
import logging
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from setfit.trainer import ColumnMappingMixin
from setfit.sampler import ContrastiveDataset
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers import InputExample, losses
from setfit.losses import SupConLoss
import gc
import json

from .training_args import TrainingArguments
from .modeling import FusionSentModel
from .merging_methods import merge_models

logging.basicConfig()
logger = logging.getLogger('FusionSent')
logger.setLevel(logging.INFO)

class Trainer(ColumnMappingMixin):
    """
    The Trainer class is responsible for managing the training and evaluation process for the FusionSent model. 

    It facilitates the training of two distinct sub-models ('setfit' and 'label_embedding') and merges their parameters
    into the unified FusionSent model. This class handles the preparation of datasets, the configuration of training parameters,
    and the execution of training and evaluation routines.
    """

    DEFAULT_EVAL_METRICS = {'metric_names': ['f1', 'precision', 'recall', 'accuracy'], 'metric_args': {'average': 'micro'}}

    def __init__(
            self,
            model: FusionSentModel = None,
            args: Optional[TrainingArguments] = TrainingArguments(),
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            eval_metrics: Optional[Dict[List,Dict]] = DEFAULT_EVAL_METRICS,
            column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:    
        """
        Initializes the Trainer class with the provided FusionSent model, training arguments, datasets, evaluation metrics, and column mapping.

        Args:
            model (FusionSentModel): The FusionSent model to be trained. If not provided, raises a RuntimeError.
            args (Optional[TrainingArguments]): Configuration for training parameters. If not provided, the default setting of 'TrainingArguments' will be used.
            train_dataset (Optional[Dataset]): The dataset used for training. If provided, applies column mapping if necessary.
            eval_dataset (Optional[Dataset]): The dataset used for evaluation. If provided, applies column mapping if necessary.
            eval_metrics (Optional[Dict[List, Dict]]): A dictionary specifying the evaluation metrics and their arguments. Defaults to evaluating f1, precision, recall, and accuracy with 'micro' averaging.
                If not provided or ill-formatted, default metrics will be used. Example format:
                    {
                        'metric_names': ['f1', 'precision', 'recall', 'accuracy'],
                        'metric_args': {'average': 'micro'}
                    } 
            column_mapping (Optional[Dict[str, str]]): A mapping of dataset columns to the expected input columns.

        Raises:
            ValueError: If the TrainingArguments are ill-formatted.
            RuntimeError: If the `model` parameter is not provided, or not of type 'FusionSentModel'.
        """

        #Verify that a model has been given.
        if model is None:
            raise ValueError("`Trainer` requires a `model` argument.")
        if not isinstance(model, FusionSentModel):
            raise ValueError("`Trainer` requires a `model` argument of type 'FusionSentModel'.")
        set_seed(12) # Seed must be set before instantiating the model when using model_init.
        self.model = model

        #Initialize 'TrainingArguments' from given input (or as default), and validate them.
        if args is not None and not isinstance(args, TrainingArguments):
            raise ValueError("`args` must be a `TrainingArguments` instance imported from `FusionSent`.")
        self.args = args
        self.args._validate()

        #Assign and validate evaluation metrics, if given.
        self.eval_metrics: Dict[List,Dict] = eval_metrics
        self._validate_eval_metrics()

        #Apply column mapping to 'train_dataset' if necessary.
        self.column_mapping = column_mapping
        if train_dataset:
            self._validate_column_mapping(train_dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the training dataset")
                train_dataset = self._apply_column_mapping(train_dataset, self.column_mapping)
        self.train_dataset = train_dataset

        #Apply column mapping to 'eval_dataset' if necessary.
        if eval_dataset:
            self._validate_column_mapping(eval_dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the evaluation dataset")
                eval_dataset = self._apply_column_mapping(eval_dataset, self.column_mapping)
        self.eval_dataset = eval_dataset


    def _dataset_to_parameters(self, dataset: Dataset) -> List[Iterable]:
        """
        Converts the provided dataset into a list of parameters required for training.

        Args:
            dataset (Dataset): The dataset to be converted.
            Expected to contain the keys 'text', 'label_description', and 'label'.

        Returns:
            List[Iterable]: A list containing three elements:
                - A list of texts from the dataset.
                - A list of label descriptions from the dataset.
                - A list of labels from the dataset.
        """
        return [dataset["text"], dataset["label_description"], dataset["label"]]

    @staticmethod
    def _has_any_multilabel(examples: List[InputExample]) -> bool:
        """
        Determines if any of the input examples represent a multi-label scenario.

        Args:
            examples (List[InputExample]): List of InputExample instances to check.

        Returns:
            bool: True if any example has a non-binary label or if any label is a list or array, False otherwise.
        """
        for example in examples:
            label = example.label

            # Check if label is a list, tuple, or numpy array (multi-label scenario)
            if isinstance(label, (list, tuple, np.ndarray)):
                return True
            
            # Check if label is not binary (i.e., not 0 or 1)
            if isinstance(label, (int, float)) and label not in {0, 1}:
                return True
        return False

    def _get_setfit_dataloader(
            self,
            x: List[str],
            y: Union[List[int], List[List[int]]],
            args: TrainingArguments,
            max_pairs: int = -1
        ) -> Tuple[DataLoader, nn.Module, int]:
        """
        Prepares a DataLoader and corresponding loss function for training the 'setfit' sub-model.

        Args:
            x (List[str]): A list of input texts.
            y (Union[List[int], List[List[int]]]): A list of binary- or multi-class labels corresponding to the input texts.
            args (TrainingArguments): The training arguments configuration.
            max_pairs (int, optional): Maximum number of pairs for contrastive sampling. Default is -1, which means no limit.

        Returns:
            Tuple[DataLoader, nn.Module, int]: A tuple containing:
                - DataLoader: The DataLoader for the 'setfit' sub-model.
                - nn.Module: The loss function for the 'setfit' sub-model.
                - int: The batch size used for the DataLoader.
        """

        # Adapt input data for sentence-transformers.
        input_data = [InputExample(texts=[text], label=label) for text, label in zip(x, y)]

        if args.setfit_loss in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            data_sampler = SentenceLabelDataset(input_data, samples_per_label=args.setfit_samples_per_label)
            batch_size = min(args.setfit_batch_size, len(data_sampler))
            dataloader = DataLoader(data_sampler, batch_size=batch_size, drop_last=True)

            if args.setfit_loss is losses.BatchHardSoftMarginTripletLoss:
                loss = args.setfit_loss(
                    model=self.model.model_body.setfit_model_body,
                    distance_metric=args.setfit_distance_metric,
                )
            elif args.setfit_loss is SupConLoss:
                loss = args.setfit_loss(model=self.model.model_body.setfit_model_body)
            else:
                loss = args.setfit_loss(
                    model=self.model.model_body.setfit_model_body,
                    distance_metric=args.setfit_distance_metric,
                    margin=args.setfit_margin,
                )
        else:
            data_sampler = ContrastiveDataset(
                examples=input_data,
                multilabel=Trainer._has_any_multilabel(input_data),
                num_iterations=args.num_iterations,
                sampling_strategy=args.setfit_sampling_strategy,
                max_pairs=max_pairs,
            )
            batch_size = min(args.setfit_batch_size, len(data_sampler))
            dataloader = DataLoader(data_sampler, batch_size=batch_size, drop_last=False)
            loss = args.setfit_loss(self.model.model_body.setfit_model_body)

        return dataloader, loss, batch_size

    def _get_label_embedding_dataloader(
            self,
            texts: List[str],
            label_descriptions: List[str],
            args: TrainingArguments
        ) -> Tuple[DataLoader, nn.Module, int]:
        """
        Prepares a DataLoader and corresponding loss function for training the 'label_embedding' sub-model.

        Note that remaining TODO's include:
        - Adding additional Sentence-Transformers losses
        - Reimplementing sampling strategies to support oversampling of negatives and undersampling of positives.

        Args:
            texts (List[str]): A list of input texts.
            label_descriptions (List[str]): A list of label descriptions corresponding to the input texts.
            args (TrainingArguments): The training arguments configuration.

        Returns:
            Tuple[DataLoader, nn.Module, int]: A tuple containing:
                - DataLoader: The DataLoader for the 'label_embedding' sub-model.
                - nn.Module: The loss function for the 'label_embedding' sub-model.
                - int: The batch size used for the DataLoader.
        """

        #TODO: add remaining ST losses
        #TODO: reimplement sampling strategies to support oversampling of negatives and undersampling of positives
        if args.label_embedding_loss is losses.MultipleNegativesRankingLoss:
            # create default dataloader with positives only
            input_data = []
            for i, text in enumerate(texts):
                for label_description in label_descriptions[i]:
                    input_data.append(InputExample(texts=[text, label_description]))

        elif args.label_embedding_loss is losses.TripletLoss:
            if args.label_embedding_sampling_strategy == "oversampling":
                # create dataloader for triplet loss with oversampling of positives
                input_data = []
                unique_labels = set([x for xs in label_descriptions for x in xs])
                for i, text in enumerate(texts):
                    negative_labels = unique_labels - set(label_descriptions[i])
                    # oversample positive label descriptions
                    positive_label_description_samples = random.choices(label_descriptions[i], k=len(negative_labels))
                    for x in range(len(negative_labels)):
                        input_data.append(InputExample(texts=[text, positive_label_description_samples[x], list(negative_labels)[x]]))
            elif args.label_embedding_sampling_strategy == "undersampling":
                # create dataloader for triplet loss with undersampling of negatives
                input_data = []
                unique_labels = set([x for xs in label_descriptions for x in xs])
                for i, text in enumerate(texts):
                    negative_labels = unique_labels - set(label_descriptions[i])
                    # undersample negative label description
                    negative_label_description_samples = random.sample(list(negative_labels), len(label_descriptions[i]))
                    for x in range(len(label_descriptions[i])):
                        input_data.append(
                            InputExample(texts=[text, label_descriptions[i][x], negative_label_description_samples[x]]))


        elif args.label_embedding_loss in [losses.ContrastiveLoss,losses.CosineSimilarityLoss,losses.OnlineContrastiveLoss]:
            if args.label_embedding_sampling_strategy == "oversampling":
                # create dataloader for contrastive learning with oversampling of positives
                input_data = []
                unique_labels =  set([x for xs in label_descriptions for x in xs])
                for i, text in enumerate(texts):
                    negative_labels = unique_labels - set(label_descriptions[i])
                    # add positive label descriptions for anchor text
                    for positive_label_description in label_descriptions[i]:
                        input_data.append(InputExample(texts=[text, positive_label_description], label=1.0))

                    # add negative label descriptions for anchor text
                    for negative_label_description in list(negative_labels):
                        input_data.append(InputExample(texts=[text, negative_label_description], label=0.0))

                    # oversample positive label descriptions
                    positive_label_description_samples = random.choices(label_descriptions[i], k=len(negative_labels)-1)
                    for positive_label_description in positive_label_description_samples:
                        input_data.append(InputExample(texts=[text, positive_label_description], label=1.0))

            elif args.label_embedding_sampling_strategy == "undersampling":
                # create dataloader for contrastive learning with undersampling of negatives
                input_data = []
                unique_labels = set([x for xs in label_descriptions for x in xs])
                for i, text in enumerate(texts):
                    negative_labels = unique_labels - set(label_descriptions[i])
                    # add positive label descriptions for anchor text
                    for positive_label_description in label_descriptions[i]:
                        input_data.append(InputExample(texts=[text, positive_label_description], label=1.0))

                    # add negative label descriptions for anchor text
                    negative_label_description_samples = random.sample(list(negative_labels),
                                                                       len(label_descriptions[i]))
                    for negative_label_description in negative_label_description_samples:
                        input_data.append(InputExample(texts=[text, negative_label_description], label=0.0))

        data_sampler = SentenceLabelDataset(input_data, samples_per_label=args.label_embedding_samples_per_label)
        batch_size = min(args.label_embedding_batch_size, len(data_sampler))
        dataloader = DataLoader(input_data, shuffle=True, batch_size=batch_size)
        loss = args.label_embedding_loss(self.model.model_body.label_embedding_model_body)

        return dataloader, loss, batch_size

    def _validate_eval_metrics(
            self,
            other: Optional[Dict[List,Dict]] = None
        ):
        """
        Validates the local evaluation metrics to ensure they contain at least one of the valid evaluation arguments: 
        `f1`, `precision`, `recall`, `accuracy`.

        Args:
            other (Optional[Dict[List,Dict]]): An alternative set of evaluation metrics to validate.

        Raises:
            ValueError: If the evaluation metrics do not contain at least one of the valid evaluation arguments.
        """
        valid_metrics = set(['f1', 'precision', 'recall', 'accuracy'])
        if other is not None and 'metric_names' in other.keys():
            provided_metrics = set(other['metric_names'])
        else:
            provided_metrics = set(self.eval_metrics.get('metric_names', []))

        if not provided_metrics.intersection(valid_metrics):
            raise ValueError(
                "'eval_metrics' did not contain at least one of the following valid values under key 'metric_names': `f1`, `precision`, `recall`, `accuracy`."
            )

    def _has_evaluation_setting(self) -> bool:
        """
        Returns a boolean indicating wether this trainer instance could perform an evaluation (has been given an evaluation dataset and metrics).
        """
        return self.eval_dataset and self.eval_metrics

    def train(
            self,
            args: Optional[TrainingArguments] = None,
            trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
            **kwargs,
    ) -> None:
        """
        This function represents the main training entry point.

        Note that evaluation will be perfomed automatically, iff a dectionary of evaluation metrics and an evaluation datatset has been provided at initialization of this instance.
        Additionally, evaluation can always be carried out manually via the 'evaluate' method.

        Args:
            args (Optional[TrainingArguments]): Training arguments to temporarily override the default training arguments for this call.
            trial (Optional[Union["optuna.Trial", Dict[str, Any]]]): The trial run or hyperparameter dictionary for hyperparameter search.

        Raises:
            ValueError: If `train_dataset` is not provided, or 'TrainingArguments' is not None and ill-formatted.
        """
        if len(kwargs):
            warnings.warn(
                f"`{self.__class__.__name__}.train` does not accept keyword arguments anymore. "
                f"Please provide training arguments via a `TrainingArguments` instance to the `{self.__class__.__name__}` "
                f"initialisation or the `{self.__class__.__name__}.train` method.",
                DeprecationWarning,
                stacklevel=2,
            )

        #Assign and validate training arguments.
        args = args or self.args or TrainingArguments()
        self.args._validate()

        #Check for existing training dataset.
        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )
        
        #Initialize trainer parameters and model for hp-search, if applicable.
        if trial:
            self._hp_search_setup(trial)

        #Construct train parameters
        train_parameters = self._dataset_to_parameters(self.train_dataset)

        #Train model body
        self._train_sentence_transformers_body(*train_parameters, args=args)

        #If evaluation dataset and metrics are given...
        if self._has_evaluation_setting():
            #...train the model head, ...
            self._train_classifier_head(
                x_train_texts=train_parameters[0],
                y_train=train_parameters[2],
                x_eval=self.eval_dataset['text'],
                y_eval=self.eval_dataset['label'],
                eval_metrics=self.eval_metrics,
                args=args
            )

            #...and evaluate the whole model.
            logger.info("  ***** Running evaluation on `eval_dataset` *****")
            self.eval_scores = self.evaluate(
                x_eval=self.eval_dataset['text'],
                y_eval=self.eval_dataset['label'],
                eval_metrics=self.eval_metrics
            )
            return self.eval_scores
        else:
            #Else, train only the model head, without evaluation.
            self._train_classifier_head(
                x_train_texts=train_parameters[0],
                y_train=train_parameters[2],
                args=args
            )
            return None

    def _train_sentence_transformers_body(
            self,
            x_train_texts: List[str],
            x_train_label_descriptions: List[List[str]],
            y_train: Optional[Union[List[int], List[List[int]]]] = None,
            args: Optional[TrainingArguments] = None
    ) -> None:
        """
        Trains both dual encoder `SentenceTransformer` bodies of the sub-models ('setfit' and 'label_embeding') for the embedding training phase.
        After training, it merges the parameters of both sub-models into the final encoder body of FusionSent.
        
        Args:
            x_train_texts (List[str]): A list of training texts.
            x_train_label_descriptions (List[List[str]]): A list of lists including label descriptions for each positive label per training text.
            y_train (Union[List[int], List[List[int]]], optional): A list of labels corresponding to the training texts.
            args (TrainingArguments, optional): Temporarily change the training arguments for this training call. If not provided, default training arguments will be used.
        
        Raises:
            ValueError: If 'args' is not None and ill-formatted.
        """
        args = args or self.args or TrainingArguments()
        args._validate()

        logger.info("  ***** Preparing training dataset *****")

        #Construct dataset for SetFit body training
        setfit_train_dataloader, setfit_loss_func, setfit_batch_size = self._get_setfit_dataloader(
            x=x_train_texts,y=y_train, args=args
        )

        #Construct dataset for label embedding training
        label_embedding_train_dataloader, label_embedding_loss_func, label_embedding_batch_size = self._get_label_embedding_dataloader(
            texts=x_train_texts, label_descriptions=x_train_label_descriptions, args=args
        )

        #Compute total number of training steps.
        setfit_total_train_steps = len(setfit_train_dataloader) * args.setfit_num_epochs
        label_embeddings_total_train_steps = len(label_embedding_train_dataloader) * args.label_embedding_num_epochs

        #Log training statistics.
        logger.info("  ***** Running sentence transformers body training *****")
        logger.info(f"  Total number of examples = {len(setfit_train_dataloader.dataset)} + {len(label_embedding_train_dataloader.dataset)}")
        logger.info(f"  Number of batches = {len(setfit_train_dataloader)} + {len(label_embedding_train_dataloader)}")
        logger.info(f"  Number of epochs = {args.setfit_num_epochs} + {args.label_embedding_num_epochs}")
        logger.info(f"  Train batch sizes = {setfit_batch_size} & {label_embedding_batch_size}")
        logger.info(f"  Total optimization steps = {setfit_total_train_steps} + {label_embeddings_total_train_steps}")

        #Train the setfit body (only if it is intended to be used).
        if args.use_setfit_body:
            setfit_warmup_steps = math.ceil(
                setfit_total_train_steps * args.setfit_warmup_proportion
            )
            self.model.model_body.setfit_model_body.fit(
                train_objectives=[(setfit_train_dataloader, setfit_loss_func)],
                epochs=args.setfit_num_epochs, warmup_steps=setfit_warmup_steps,
                show_progress_bar=args.show_progress_bar
            )
            setfit_loss_func.to('cpu')
            self.model.model_body.setfit_model_body.to('cpu')
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

        #Train the label_embeddings body.
        label_embeddings_warmup_steps = math.ceil(
            label_embeddings_total_train_steps * args.label_embedding_warmup_proportion
        )
        self.model.model_body.label_embedding_model_body.fit(
            train_objectives=[(label_embedding_train_dataloader, label_embedding_loss_func)],
            epochs=args.label_embedding_num_epochs,
            warmup_steps=label_embeddings_warmup_steps,
            show_progress_bar=args.show_progress_bar
        )
        label_embedding_loss_func.to('cpu')
        self.model.model_body.label_embedding_model_body.to('cpu')
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        #Get parameters of both trained models
        setfit_parameter_dict = dict(
            self.model.model_body.setfit_model_body._first_module().auto_model.named_parameters()
        )
        label_embedding_parameter_dict = dict(
            self.model.model_body.label_embedding_model_body._first_module().auto_model.named_parameters()
        )

        #Fuse/merge model parameters with selected algorithm.
        t = 0.5 if args.use_setfit_body else 0
        fused_parameter_dict = merge_models(
            model_state_dict0=label_embedding_parameter_dict,
            model_state_dict1=setfit_parameter_dict,
            t=t,
            merging_method=args.merging_method
        )

        #Initialize the body of the final FusionSent model with the fused model parameters.
        fusion_state_dict = self.model.model_body.fusion_model_body._first_module().auto_model.state_dict()
        for key in fusion_state_dict:
            fusion_state_dict[key] = fused_parameter_dict[key]
        self.model.model_body.fusion_model_body._first_module().auto_model.load_state_dict(fusion_state_dict)

    @staticmethod
    def _ensure_single_label_format(labels: Union[List[int], List[List[int]]]):
        """
        Helper function to convert a list of labels into single-label format, if neccesary.
        """
        if isinstance(labels[0], list):
            return np.argmax(labels, axis=1)
        return labels

    def _train_classifier_head(
            self,
            x_train_texts: List[str],
            y_train: Union[List[int], List[List[int]]],
            x_eval: Optional[List[str]] = None,
            y_eval: Optional[List[int]] = None,
            eval_metrics: Optional[Dict[List, Dict]] = None,
            args: Optional[TrainingArguments] = None,
    ) -> None:
        """
        Trains a classification head for each candidate model body (`setfit`, `label_embedding`, and their 'fusion').
        If evaluation metrics and dataset are provided, the performance of all final models will be evaluted and the best performing model is set as the default for further use.

        Note: Cross-Validation is yet to be implemented.

        Args:
            x_train_texts (List[str]): A list of training texts.
            y_train (Union[List[int], List[List[int]]]): A list of labels corresponding to the training texts.
            x_eval (Optional[List[str]]): A list of evaluation texts.
            y_eval (Optional[List[int]]): A list of labels corresponding to the evaluation texts.
            eval_metrics (Optional[Dict[List, Dict]]): A dictionary specifying the evaluation metrics and their respective arguments.
                If not provided, evaluation will be omitted. If ill-formatted, default metrics will be used. Example format:
                    {
                        'metric_names': ['f1', 'precision', 'recall', 'accuracy'],
                        'metric_args': {'average': 'micro'}
                    } 
            args (Optional[TrainingArguments]): Training arguments to temporarily override the default training arguments for this call.
        """
        logger.info("  ***** Running classification head training *****")
        y_train = Trainer._ensure_single_label_format(y_train) # Necessary for model head trainig.

        #Get embeddings from setfit body and train setfit model head.
        self.model.set_prediction_strategy("setfit")
        setfit_train_features = self.model.model_body.setfit_model_body.encode(x_train_texts, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.model_head.setfit_model_head.fit(setfit_train_features, y_train)

        #Get embeddings from label_embedding body and train label_embedding model head.
        self.model.set_prediction_strategy("label_embedding")
        label_embedding_train_features = self.model.model_body.label_embedding_model_body.encode(x_train_texts, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.model_head.label_embedding_model_head.fit(label_embedding_train_features, y_train)

        #Get embeddings from fusion body and train fusion model head.
        self.model.set_prediction_strategy("fusion")
        fusion_train_features = self.model.model_body.fusion_model_body.encode(x_train_texts, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.model_head.fusion_model_head.fit(fusion_train_features, y_train)

        #Evaluate classifications with different body features and set the best performing one as default.
        eval_dict = {}
        if x_eval and y_eval and eval_metrics:
        
            #Use evaluation dataset to choose best performing features.
            self.model.set_prediction_strategy("setfit")
            setfit_eval_scores = self.evaluate(x_eval=x_eval, y_eval=y_eval, eval_metrics=eval_metrics)
            print("SetFit eval scores:", setfit_eval_scores)
            eval_dict["SetFit eval scores"] = setfit_eval_scores
            self.model.set_prediction_strategy("label_embedding")
            label_embedding_eval_scores = self.evaluate(x_eval=x_eval, y_eval=y_eval, eval_metrics=eval_metrics)
            print("Label embedding eval scores:", label_embedding_eval_scores)
            eval_dict["Label embedding eval scores"] = label_embedding_eval_scores
            self.model.set_prediction_strategy("fusion")
            fusion_eval_scores = self.evaluate(x_eval=x_eval, y_eval=y_eval, eval_metrics=eval_metrics)
            print("Fusion eval scores:", fusion_eval_scores)
            eval_dict["Fusion eval scores"] = fusion_eval_scores

            #Save evaluation dictionary, if path was provided.
            if args.json_path is not None:
                with open(args.json_path + '.json', 'w') as fp:
                        json.dump(eval_dict, fp)

            #choose best performing model from average of evaluation scores as default mode.
            mean_eval_scores = {}
            mean_eval_scores['fusion'] = np.mean(list(fusion_eval_scores.values()))
            mean_eval_scores['label_embedding'] = np.mean(list(label_embedding_eval_scores.values()))
            mean_eval_scores['setfit'] = np.mean(list(setfit_eval_scores.values()))
            self.model.set_prediction_strategy(max(mean_eval_scores, key=mean_eval_scores.get))

        else:
            #TODO: Perform cross-validation to get best performing features
            pass

    def evaluate(
            self, x_eval: List[str],
            y_eval: Union[List[int], List[List[int]]],
            eval_metrics: Optional[Dict[List,Dict]] = None
        ):
        """
        Evaluates the performance of the full model on a given evaluation dataset.
        Note that this depends on the model's current prediction_strategy (i.e. which encoder body it will use) to perform inference.

        Args:
            x_eval (List[str]): A list of evaluation texts.
            y_eval (Union[List[int], List[List[int]]]): A list of labels corresponding to the evaluation texts.
            eval_metrics (Optional[Dict[List, Dict]]): A dictionary specifying the evaluation metrics and their respective arguments, to temporarily override the default (if any).
                If not provided or ill-formatted, default metrics will be used. Example format:
                {
                    'metric_names': ['f1', 'precision', 'recall', 'accuracy'],
                    'metric_args': {'average': 'micro'}
                }. 

        Returns:
            Dict[str, float]: A dictionary containing the computed scores for each specified metric.
       """
        
        #If no eval_metrics were given, use the configured ones.
        if not eval_metrics:
            eval_metrics = self.eval_metrics

        #Validate eval_metrics and use the default if ill-formatted.
        try:
            self._validate_eval_metrics(eval_metrics)
        except ValueError as e:
            if "eval_metrics" in str(e):
                eval_metrics = self.DEFAULT_EVAL_METRICS
                warnings.warn(
                    "'eval_metrics' provided were ill-formatted. Falling back to default metrics.",
                    UserWarning,
                    stacklevel=2,
                )

        #Perform inference on the evaluation dataset.
        y_pred = self.model.predict(x_eval)
        y_true = Trainer._ensure_single_label_format(y_eval)

        #Correctly format eval_metrics if only a single one was given.
        if isinstance(eval_metrics['metric_names'], str):
            eval_metrics['metric_names'] = [eval_metrics['metric_names']]

        #Perform the evaluation.
        eval_scores = {}
        for metric in eval_metrics['metric_names']:
            if metric == "f1":
                eval_scores["f1"] = f1_score(y_true, y_pred, average=eval_metrics['metric_args']['average'])
            elif metric == "precision":
                eval_scores["precision"] = precision_score(y_true, y_pred, average=eval_metrics['metric_args']['average'])
            elif metric == "recall":
                eval_scores["recall"] = recall_score(y_true, y_pred, average=eval_metrics['metric_args']['average'])
            elif metric == "accuracy":
                eval_scores["accuracy"] = accuracy_score(y_true, y_pred)

        #Check wether evalaution metrics are present. Note: This must always be the case, if a successful validation has occured (so in theory, this exception should never be raised).
        if not eval_scores:
            raise ValueError(
                "eval_metrics did not contain at least on of the following valid evaluation arguments: `f1`, `precision`, `recall`, `accuracy`."
                )
        
        return eval_scores