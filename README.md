# xs2arr

`xs2arr` computes the Arrhenius coefficients for reactions with provided cross section data,
$\sigma\left(\epsilon\right)$. For a range of temperatures, $T_{\text{eff}}$, a defined electron energy distribution
function (EEDF) is computed,
```math
f_{T_{\text{eff}}}\left(\epsilon\right)=\beta_{1}\left(\alpha\,T_{\text{eff}}\right)^{-\alpha}\sqrt{\epsilon}
\:\exp{\left[-\frac{\epsilon^{x}\beta_{2}}{\alpha\,T_{\text{eff}}}\right]},
```
where $\alpha$, $\beta_{1}$ and $\beta_{2}$ are determinable parameters. A Maxwellian is defined with $x=1$ and a
Druyvesteyn with $x=2$.

The reaction rate coefficient, $k\left(T_{\text{eff}}\right)$, is then calculated via an integral,
```math
k\left({T_{\text{eff}}}\right)=\sqrt{\frac{2q}{m_{e}}}\int_{0}^{\infty}\sqrt{\epsilon}\:\sigma\left(\epsilon\right)
f_{T_{\text{eff}}}\left(\epsilon\right)\mathrm{d}\epsilon.
```
where $q$ and $m_{e}$ are the charge and mass of the electron, respectively. The cross section data is interpolated onto
the EEDF grid.

The $T_{\text{eff}}$ and $k\left({T_{\text{eff}}}\right)$ data are then fit to the modified Arrhenius equation,
```math
k\left({T_{\text{eff}}}\right)=a\,T_{\text{eff}}^{b}\:\exp\left(\frac{c}{T_{\text{eff}}}\right).
```
to determine the constants, $a$, $b$ and $c$.

**See the provided Jupyter notebook in `notebooks/xs2arr_guide.ipynb` for a detailed guide through the code.**


## Usage

In a Python file, a `Model` can be created and fitted,
```python
from xs2arr import Model

model = Model(lxcat_file="path/to/input/file.txt")

results = model.fit()

fitting, a, b, c = results[-1]
```
The `fitting` parameter returned is a `ModelResult`
[object](https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult) from the `lmfit` package. This gives
information on the performance of the fitting, as well as other computed quantities.

There is a choice of EEDF, as well as a definable grid and choice of integrator,
```python
from numpy import linspace

eedf_grid = linspace(start=0.0, stop=100.0, num=10000, dtype=float)

model = Model(lxcat_file,
              eedf_type="maxwellian",
              eedf_grid=eedf_grid,
              integrator="simpson")
```
And the fitting can be provided a defined temperature grid, as well as the choice of the fitting being performed in
logarithmic space,
```python
T_grid = linspace(start=0.001, stop=6.0, num=1000, dtype=float)

results = model.fit(T_grid=T_grid, logarithmic=True)
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License

[MIT](https://choosealicense.com/licenses/mit/)