"""
referencef.registry
===================

Registry system for simulation data generators.

The registry provides a lightweight mechanism for registering and
retrieving generator classes by name.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from typing import Any, Type

from .generators import DataGenerator


class GeneratorRegistry:
    """
    Registry of available data generator classes.

    Examples
    --------
    >>> GeneratorRegistry.register("normal", NormalGenerator)
    >>> generator = GeneratorRegistry.create(
    ...     "normal",
    ...     config=config,
    ...     means=[0, 0],
    ...     std=[1, 1],
    ... )
    """

    _registry: dict[str, Type[DataGenerator]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        generator: Type[DataGenerator],
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a generator class.

        Parameters
        ----------
        name:
            Name used to retrieve the generator.

        generator:
            Generator class.

        overwrite:
            Whether to replace an existing registration.

        Raises
        ------
        ValueError
            If the name already exists and overwrite is False.
        """

        key = cls._normalize_name(name)

        if key in cls._registry and not overwrite:
            raise ValueError(
                f"Generator '{key}' is already registered."
            )

        if not issubclass(generator, DataGenerator):
            raise TypeError(
                "Generator must inherit from DataGenerator."
            )

        cls._registry[key] = generator

    @classmethod
    def unregister(
        cls,
        name: str,
    ) -> None:
        """
        Remove a generator from the registry.

        Raises
        ------
        KeyError
            If the generator is not registered.
        """

        key = cls._normalize_name(name)

        if key not in cls._registry:
            raise KeyError(
                f"Generator '{key}' is not registered."
            )

        del cls._registry[key]

    @classmethod
    def get(
        cls,
        name: str,
    ) -> Type[DataGenerator]:
        """
        Retrieve a generator class.

        Raises
        ------
        KeyError
            If the generator does not exist.
        """

        key = cls._normalize_name(name)

        try:
            return cls._registry[key]

        except KeyError as exc:

            available = ", ".join(cls.available())

            raise KeyError(
                f"Unknown generator '{key}'. "
                f"Available generators: {available}"
            ) from exc

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> DataGenerator:
        """
        Instantiate a registered generator.

        Parameters
        ----------
        name:
            Registered generator name.

        kwargs:
            Arguments passed to the generator constructor.

        Returns
        -------
        DataGenerator
            Instantiated generator.
        """

        generator_class = cls.get(name)

        return generator_class(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """
        Return registered generator names.
        """

        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Remove all registrations.

        Primarily useful for testing.
        """

        cls._registry.clear()

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize registry keys.
        """

        if not isinstance(name, str):
            raise TypeError(
                "Generator name must be a string."
            )

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError(
                "Generator name cannot be empty."
            )

        return normalized