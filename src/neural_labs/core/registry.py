from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import nn

T = TypeVar("T")


class RegistryError(LookupError):
    pass


@dataclass(frozen=True)
class RegisteredFactory:
    name: str
    factory: Callable[..., Any]
    domain: str
    description: str = ""


class FactoryRegistry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, RegisteredFactory] = {}

    def register(
        self,
        name: str,
        *,
        domain: str,
        description: str = "",
        replace: bool = False,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items and not replace:
                raise RegistryError(f"{self.kind} ya registrado: {name}")
            self._items[name] = RegisteredFactory(name, factory, domain, description)
            return factory

        return decorator

    def create(self, name: str, /, **kwargs: Any) -> Any:
        try:
            item = self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items))
            raise RegistryError(f"{self.kind} desconocido {name!r}. Disponibles: {available}") from exc
        return item.factory(**kwargs)

    def get(self, name: str) -> RegisteredFactory:
        try:
            return self._items[name]
        except KeyError as exc:
            raise RegistryError(f"{self.kind} desconocido: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": item.name, "domain": item.domain, "description": item.description}
            for item in sorted(self._items.values(), key=lambda item: item.name)
        ]


MODEL_REGISTRY = FactoryRegistry("modelo")
EXPERIMENT_REGISTRY = FactoryRegistry("experimento")


def ensure_module(model: Any) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError(f"La factoría devolvió {type(model).__name__}, se esperaba torch.nn.Module")
    return model


def module_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def first_parameter_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")
