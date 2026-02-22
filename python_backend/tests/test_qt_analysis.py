import pytest
from analysis.qt_analysis import QTIntervalAnalyzer

@pytest.fixture
def analyzer():
    return QTIntervalAnalyzer()

def test_qtc_calculations(analyzer):
    qt = 400 # ms
    hr = 60 # bpm -> RR = 1.0s
    
    # At RR=1.0, all formulas should equal QT
    assert analyzer.calculate_qtc(qt, 1.0, 'bazett') == 400
    assert analyzer.calculate_qtc(qt, 1.0, 'fridericia') == 400
    
    # High HR (100bpm -> RR=0.6s)
    # Bazett: 400 / sqrt(0.6) = 400 / 0.77 = 516
    # Fridericia: 400 / cbrt(0.6) = 400 / 0.84 = 476
    qtc_b = analyzer.calculate_qtc(qt, 0.6, 'bazett')
    qtc_f = analyzer.calculate_qtc(qt, 0.6, 'fridericia')
    
    assert qtc_b > qtc_f # Bazett overcorrects high HR

def test_qt_dispersion(analyzer):
    measurements = {'V1': 380, 'V2': 400, 'V5': 440}
    res = analyzer.calculate_dispersion(measurements)
    
    assert res['qt_max'] == 440
    assert res['qt_min'] == 380
    assert res['qt_dispersion'] == 60

def test_tdp_risk_high(analyzer):
    # QTc > 500ms
    res = analyzer.assess_tdp_risk(510, 30, 'Male')
    assert res['risk_level'] in ["Moderate", "Very High"]
    assert "Severe QTc prolongation" in res['risk_factors'][0]

def test_full_analysis(analyzer):
    qt_leads = {'II': 400, 'V5': 410}
    res = analyzer.analyze(qt_leads, hr=60, qrs_ms=100, sex='Female')
    
    assert 'qtc_primary_ms' in res
    assert 'tdp_risk' in res
    assert res['qt_interval_ms'] == 410 # Max of inputs
