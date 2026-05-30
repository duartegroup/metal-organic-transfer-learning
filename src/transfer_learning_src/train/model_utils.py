import torch
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelComponentManager:
    """
    Utility class for managing model component saving and loading.

    This class provides static methods for saving and loading individual components
    of neural network models (e.g., encoders, classification heads) with proper
    error handling and metadata tracking. Supports both standalone models and
    wrapped models (e.g., CCSA wrappers).
    """

    # Standard component names and their typical attributes in models
    COMPONENT_MAPPING = {
        "mol_encoder": "mol_encoder",
        "protein_encoder": "protein_encoder",
        "pocket_encoder": "pocket_encoder",
        "feature_extractor": "feature_extractor",
        "classification_head": "classification_head",
        "projection_head": "projection_head",
    }

    @staticmethod
    def get_saveable_components(model: torch.nn.Module) -> Dict[str, str]:
        """
        Get a mapping of component names to their actual attribute paths in the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to extract component mappings from. Can be a base model or
            a wrapped model (e.g., CCSAFinetune).

        Returns
        -------
        Dict[str, str]
            Dictionary mapping component names (e.g., 'mol_encoder') to their
            attribute paths in the model (e.g., 'base_model.mol_encoder' or 'mol_encoder').
        """
        components = {}

        # Handle CCSA wrapper - check if model has base_model attribute
        base_model = getattr(model, "base_model", model)
        has_wrapper = hasattr(model, "base_model")

        # Check each component
        for (
            component_name,
            attr_name,
        ) in ModelComponentManager.COMPONENT_MAPPING.items():
            if hasattr(base_model, attr_name):
                component = getattr(base_model, attr_name)
                if component is not None:
                    # Determine the correct path for accessing this component
                    if has_wrapper:
                        components[component_name] = f"base_model.{attr_name}"
                    else:
                        components[component_name] = attr_name

        return components

    @staticmethod
    def save_components(
        model: torch.nn.Module, output_dir: Path, phase_name: str
    ) -> Dict[str, str]:
        """
        Save all model components to disk with metadata.

        This method saves each model component (encoders, heads) as separate .pth files
        and creates a metadata JSON file containing information about the saved components.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose components should be saved.
        output_dir : Path
            Directory where component files and metadata will be saved.
        phase_name : str
            Name of the training phase (e.g., 'pretrain', 'ccsa', 'finetune').

        Returns
        -------
        Dict[str, str]
            Dictionary mapping component names to their saved file paths.
        """
        components = ModelComponentManager.get_saveable_components(model)

        if not components:
            logger.warning(f"No saveable components found for {phase_name}")
            return {}

        saved_components = {}
        component_metadata = {
            "phase": phase_name,
            "model_type": model.__class__.__name__,
            "has_wrapper": hasattr(model, "base_model"),
            "components": {},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        try:
            for component_name, component_path in components.items():
                # Navigate to the actual component
                component_obj = model
                for attr in component_path.split("."):
                    component_obj = getattr(component_obj, attr)

                # Save the component state dict
                component_file = output_dir / f"{component_name}.pth"
                torch.save(component_obj.state_dict(), component_file)
                saved_components[component_name] = str(component_file)

                # Store component metadata
                component_metadata["components"][component_name] = {
                    "file_path": str(component_file),
                    "model_path": component_path,
                    "component_class": component_obj.__class__.__name__,
                    "num_parameters": sum(
                        p.numel() for p in component_obj.parameters()
                    ),
                }

                logger.debug(f"Saved {component_name} to {component_file}")

            # Save metadata
            metadata_file = output_dir / "component_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(component_metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving components for {phase_name}: {e}")
            return {}

        logger.info(
            f"Saved {len(saved_components)} components for {phase_name}: {list(saved_components.keys())}"
        )
        return saved_components

    @staticmethod
    def load_components_from_directory(
        model: torch.nn.Module,
        load_dir: Path,
        strict: bool = False,
        components_to_load: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Load saved model components from a directory into a model.

        This method loads previously saved component state dicts from disk and applies
        them to the corresponding components in the target model. It supports partial
        loading (loading only specific components) and flexible parameter matching.

        Parameters
        ----------
        model : torch.nn.Module
            Target model to load components into.
        load_dir : Path
            Directory containing saved component files (.pth) and metadata.
        strict : bool, optional
            Whether to require exact parameter matching when loading, by default False.
        components_to_load : Optional[List[str]], optional
            Specific component names to load. If None, attempts to load all available
            components, by default None.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping component names to loading success status (True if loaded
            successfully, False otherwise).

        Raises
        ------
        FileNotFoundError
            If the specified load directory does not exist.
        """
        load_dir = Path(load_dir)

        # Check if directory exists
        if not load_dir.exists():
            raise FileNotFoundError(f"Load directory does not exist: {load_dir}")

        # Load metadata if available
        metadata_file = load_dir / "component_metadata.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded component metadata from {metadata_file}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        # Get available component files
        available_components = {}
        for component_name in ModelComponentManager.COMPONENT_MAPPING.keys():
            component_file = load_dir / f"{component_name}.pth"
            if component_file.exists():
                available_components[component_name] = component_file

        if not available_components:
            logger.warning(f"No component files found in {load_dir}")
            return {}

        # Determine which components to load
        if components_to_load is None:
            components_to_load = list(available_components.keys())
        else:
            # Filter to only available components
            components_to_load = [
                c for c in components_to_load if c in available_components
            ]

        logger.info(f"Loading components: {components_to_load}")

        # Load each component
        load_status = {}
        target_components = ModelComponentManager.get_saveable_components(model)

        for component_name in components_to_load:
            try:
                if component_name not in target_components:
                    logger.warning(
                        f"Component '{component_name}' not found in target model"
                    )
                    load_status[component_name] = False
                    continue

                # Load the component state dict
                component_file = available_components[component_name]
                component_state = torch.load(
                    component_file, map_location="cpu", weights_only=True
                )

                # Navigate to the target component in the model
                target_component = model
                component_path = target_components[component_name]
                for attr in component_path.split("."):
                    target_component = getattr(target_component, attr)

                # Load the state dict
                if strict:
                    target_component.load_state_dict(component_state)
                else:
                    # Use strict=False to allow partial loading
                    missing_keys, unexpected_keys = target_component.load_state_dict(
                        component_state, strict=False
                    )
                    if missing_keys:
                        logger.debug(
                            f"Missing keys in {component_name}: {missing_keys}"
                        )
                    if unexpected_keys:
                        logger.debug(
                            f"Unexpected keys in {component_name}: {unexpected_keys}"
                        )

                load_status[component_name] = True
                logger.debug(
                    f"Successfully loaded {component_name} from {component_file}"
                )

            except Exception as e:
                logger.error(f"Failed to load component '{component_name}': {e}")
                load_status[component_name] = False

        successful_loads = sum(load_status.values())
        logger.info(
            f"Successfully loaded {successful_loads}/{len(components_to_load)} components"
        )

        return load_status

    @staticmethod
    def load_specific_components(
        model: torch.nn.Module, component_paths: Dict[str, str], strict: bool = False
    ) -> Dict[str, bool]:
        """
        Load specific model components from individual file paths.

        This method allows loading components from arbitrary file paths rather than
        from a standard directory structure.

        Parameters
        ----------
        model : torch.nn.Module
            Target model to load components into.
        component_paths : Dict[str, str]
            Dictionary mapping component names to their file paths.
        strict : bool, optional
            Whether to require exact parameter matching when loading, by default False.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping component names to loading success status.
        """
        load_status = {}
        target_components = ModelComponentManager.get_saveable_components(model)

        for component_name, file_path in component_paths.items():
            try:
                if component_name not in target_components:
                    logger.warning(
                        f"Component '{component_name}' not found in target model"
                    )
                    load_status[component_name] = False
                    continue

                # Load component state
                component_state = torch.load(
                    file_path, map_location="cpu", weights_only=True
                )

                # Navigate to target component
                target_component = model
                component_path = target_components[component_name]
                for attr in component_path.split("."):
                    target_component = getattr(target_component, attr)

                # Load state dict
                if strict:
                    target_component.load_state_dict(component_state)
                else:
                    target_component.load_state_dict(component_state, strict=False)

                load_status[component_name] = True
                logger.debug(f"Successfully loaded {component_name} from {file_path}")

            except Exception as e:
                logger.error(
                    f"Failed to load component '{component_name}' from {file_path}: {e}"
                )
                load_status[component_name] = False

        return load_status

    @staticmethod
    def list_available_components(directory: Path) -> Dict[str, Any]:
        """
        List all available model components in a directory.

        This method scans a directory for saved component files and returns
        information about each available component, including metadata if present.

        Parameters
        ----------
        directory : Path
            Directory to search for component files.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping component names to their information, including:
            - 'file_path': Path to the component file
            - 'metadata': Component metadata if available (from metadata JSON)
        """
        directory = Path(directory)

        if not directory.exists():
            return {}

        # Load metadata if available
        metadata_file = directory / "component_metadata.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            except:
                pass

        # Find available component files
        available = {}
        for component_name in ModelComponentManager.COMPONENT_MAPPING.keys():
            component_file = directory / f"{component_name}.pth"
            if component_file.exists():
                available[component_name] = {
                    "file_path": str(component_file),
                    "file_size": component_file.stat().st_size,
                    "metadata": metadata.get("components", {}).get(component_name, {}),
                }

        return {
            "directory": str(directory),
            "phase": metadata.get("phase", "unknown"),
            "model_type": metadata.get("model_type", "unknown"),
            "timestamp": metadata.get("timestamp", "unknown"),
            "components": available,
        }
