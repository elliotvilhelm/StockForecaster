from market import EquityData
import matplotlib.pyplot as plt

def test_equity_loads_data():
    e = EquityData('data/SPY.csv', 'SPY')
    assert(type(e.data) == dict)
