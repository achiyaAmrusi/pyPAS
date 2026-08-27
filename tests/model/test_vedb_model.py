import pytest
from scipas.model import Sample, Material, Layer


@pytest.fixture
def material()->Material:
    silicon = Material(name="Silicon",diffusion = 1,mobility = 1,bulk_annihilation_rate = 1)
    return silicon


@pytest.fixture
def layer(material)->Layer:
    layer = Layer(start=0.0, width=10000.0, material=material)
    return layer


@pytest.fixture
def one_layer_sample(layer)->Sample:
    sample = Sample(layers=[layer], absorption_length=1)
    return sample

# ── check the model properties ───────────────────────────────────────────────────────
