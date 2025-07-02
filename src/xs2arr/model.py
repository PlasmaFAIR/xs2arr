from xs2arr.constants import (
    druyvesteyn_beta1,
    druyvesteyn_beta2,
    maxwellian_beta1,
    maxwellian_beta2,
)
from xs2arr.io import parse_lxcat_data


class Model:
    def __init__(
        self,
        lxcat_file: str | None = None,
        eedf_type: str = "maxwellian",
    ):
        if lxcat_file is None:
            raise ValueError("No lxcat file provided")
        if not isinstance(lxcat_file, str):
            raise TypeError("lxcat_file must be of type str")
        if not lxcat_file:
            raise ValueError("lxcat_file cannot be an empty string")

        self.cross_section_set = parse_lxcat_data(lxcat_file)

        self._eedf_beta1 = 0.0
        self._eedf_beta2 = 0.0
        self._eedf_type = None
        self.eedf_type = eedf_type

    @property
    def eedf_type(self) -> str:
        """Which EEDF to use: 'maxwellian' or 'druyvesteyn'."""
        return self._eedf_type

    @eedf_type.setter
    def eedf_type(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("eedf_type must be of type str")
        if value not in ("maxwellian", "druyvesteyn"):
            raise ValueError("eedf_type must be 'maxwellian' or 'druyvesteyn'")

        self._eedf_type = value

        self._set_eedf_constants()

    def _set_eedf_constants(self):
        """Sets the constants beta1 and beta2 for the EEDF based on its type."""
        if self._eedf_type == "maxwellian":
            self._eedf_beta1 = maxwellian_beta1
            self._eedf_beta2 = maxwellian_beta2
        elif self._eedf_type == "druyvesteyn":
            self._eedf_beta1 = druyvesteyn_beta1
            self._eedf_beta2 = druyvesteyn_beta2
        else:
            raise ValueError("self._eedf_type is set incorrectly")
