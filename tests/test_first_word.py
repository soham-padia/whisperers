"""first_word reads the answer even when the reply glues it to junk or opens with a stray symbol."""
import pytest

from whisperers.tasks import first_word


@pytest.mark.parametrize("reply, word", [
    (" cold", "cold"),
    (" Cold.\nhot -> cold", "cold"),
    ("> cold ! !", "cold"),                 # stray symbol first (scored wrong before the fix)
    (" disruption●394 to be", "disruption"),  # answer glued to junk
    (" appreciation_ _ _ _", "appreciation"),
    (" anti-inflammatory", "anti-inflammatory"),
    (" don't", "don't"),
    (" Frío", "frío"),
    (" 75900\n", "75900"),                  # numbers still read as before
    (" ! ! !", ""),
    ("", ""),
])
def test_first_word(reply, word):
    assert first_word(reply) == word
