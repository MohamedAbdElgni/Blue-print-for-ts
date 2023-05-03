Blue print for time series models ARIMA, AUTOReg, garch.
===========================
## Installation

```bash 
pip install -r requirements.txt
```

## Example

```python

    from pred_ts import Model
    df=pd.series()       
    x=Model(data=df,freq='d',returns=False)
    x.eda()
    x.arima_mod_grid(p_param=range(0, 25, 8), q_param=range(0, 3, 1),cut=0.8)
    x.arima(order=(8,0,0),wfv=True)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Project Status

This project is still under development and this is the very beta version,

only ARIMA model is implemented.

## Contact:
- [Email](mailto:mohamed.a.abdelgani@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/mohamedahmed878/)
- [Kaggle](https://www.kaggle.com/mohamedahmed878)
Enjoy :yum:
