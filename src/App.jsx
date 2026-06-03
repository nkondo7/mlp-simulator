import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

function pseudoRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function normalLikeNoise(seed) {
  let sum = 0;
  for (let i = 0; i < 12; i++) {
    sum += pseudoRandom(seed * 100 + i);
  }
  return sum - 6;
}

const DATA_PATTERNS = [
  {
    id: "smooth",
    label: "A：なめらか",
    shortLabel: "なめらか",
    recommendedUnits: 2,
    noiseScale: 0.45,
    yMin: -5.5,
    yMax: 5.5,
    fn: (x) => 2.0 * Math.sin(1.25 * x + 0.3) + 0.45 * x,
  },
  {
    id: "two_bumps",
    label: "B：山と谷",
    shortLabel: "山と谷",
    recommendedUnits: 3,
    noiseScale: 0.55,
    yMin: -6.5,
    yMax: 6.5,
    fn: (x) =>
      1.45 * Math.sin(1.2 * x) +
      1.7 * Math.exp(-1.1 * (x - 1.15) ** 2) -
      1.45 * Math.exp(-1.55 * (x + 1.25) ** 2) +
      0.12 * x ** 3,
  },
  {
    id: "wavy",
    label: "C：波が多い",
    shortLabel: "波が多い",
    recommendedUnits: 5,
    noiseScale: 0.5,
    yMin: -11,
    yMax: 11,
    fn: (x) => 3.4 * Math.sin(1.3 * x) + 2.15 * Math.sin(3.15 * x + 0.5) + 0.55 * x,
  },
  {
    id: "hard",
    label: "D：複雑",
    shortLabel: "複雑",
    recommendedUnits: 6,
    noiseScale: 0.6,
    yMin: -12,
    yMax: 12,
    fn: (x) =>
      3.9 * Math.sin(1.35 * x) +
      2.05 * Math.sin(3.2 * x + 0.4) +
      2.55 * Math.exp(-1.15 * (x - 1.25) ** 2) -
      2.2 * Math.exp(-1.8 * (x + 1.45) ** 2) +
      0.34 * x ** 3 -
      0.55 * x,
  },
];

const MAX_HIDDEN_UNITS = 6;
const UNIT_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"];

function makeDefaultParams() {
  return {
    outputBias: 0,
    units: Array.from({ length: MAX_HIDDEN_UNITS }, (_, j) => ({
      inputWeight: 0.8 + j * 0.45,
      hiddenBias: j % 2 === 0 ? -0.6 : 0.6,
      outputWeight: 0,
    })),
  };
}

function cloneParams(params) {
  return {
    outputBias: params.outputBias,
    units: params.units.map((unit) => ({ ...unit })),
  };
}

function activeUnits(params, hiddenUnitCount) {
  return params.units.slice(0, hiddenUnitCount);
}

function predictMlp(params, x, hiddenUnitCount) {
  let y = params.outputBias;
  for (const unit of activeUnits(params, hiddenUnitCount)) {
    y += unit.outputWeight * Math.tanh(unit.inputWeight * x + unit.hiddenBias);
  }
  return y;
}

function unitActivation(unit, x) {
  return Math.tanh(unit.inputWeight * x + unit.hiddenBias);
}

function unitContribution(unit, x) {
  return unit.outputWeight * unitActivation(unit, x);
}

function calculateRmse(data, params, hiddenUnitCount) {
  if (data.length === 0) return 0;
  const mse =
    data.reduce((sum, point) => {
      const yPred = predictMlp(params, point.x, hiddenUnitCount);
      return sum + (point.y - yPred) ** 2;
    }, 0) / data.length;
  return Math.sqrt(mse);
}

function ensureTrainableParams(params, hiddenUnitCount, seed = 1) {
  const next = cloneParams(params);
  for (let j = 0; j < hiddenUnitCount; j++) {
    if (Math.abs(next.units[j].inputWeight) < 1e-6) {
      next.units[j].inputWeight = (pseudoRandom(seed * 300 + j * 5 + 1) - 0.5) * 2;
    }
    if (Math.abs(next.units[j].outputWeight) < 1e-6) {
      next.units[j].outputWeight = (pseudoRandom(seed * 300 + j * 5 + 2) - 0.5) * 0.9;
    }
  }
  return next;
}

function gradientStep(data, currentParams, hiddenUnitCount, options = {}) {
  const learningRate = options.learningRate ?? 0.022;
  const l2 = options.l2 ?? 0.0006;
  const params = cloneParams(currentParams);
  const n = data.length;

  let gradOutputBias = 0;
  const gradUnits = Array.from({ length: hiddenUnitCount }, () => ({
    inputWeight: 0,
    hiddenBias: 0,
    outputWeight: 0,
  }));

  for (const point of data) {
    const activations = [];
    let yPred = params.outputBias;

    for (let j = 0; j < hiddenUnitCount; j++) {
      const unit = params.units[j];
      const h = Math.tanh(unit.inputWeight * point.x + unit.hiddenBias);
      activations[j] = h;
      yPred += unit.outputWeight * h;
    }

    const error = yPred - point.y;
    const common = (2 * error) / n;
    gradOutputBias += common;

    for (let j = 0; j < hiddenUnitCount; j++) {
      const unit = params.units[j];
      const h = activations[j];
      const tanhDerivative = 1 - h * h;
      gradUnits[j].outputWeight += common * h;
      gradUnits[j].inputWeight += common * unit.outputWeight * tanhDerivative * point.x;
      gradUnits[j].hiddenBias += common * unit.outputWeight * tanhDerivative;
    }
  }

  params.outputBias -= learningRate * gradOutputBias;
  params.outputBias = clamp(params.outputBias, -6, 6);

  for (let j = 0; j < hiddenUnitCount; j++) {
    const unit = params.units[j];
    unit.inputWeight -= learningRate * (gradUnits[j].inputWeight + l2 * unit.inputWeight);
    unit.hiddenBias -= learningRate * gradUnits[j].hiddenBias;
    unit.outputWeight -= learningRate * (gradUnits[j].outputWeight + l2 * unit.outputWeight);

    unit.inputWeight = clamp(unit.inputWeight, -7, 7);
    unit.hiddenBias = clamp(unit.hiddenBias, -7, 7);
    unit.outputWeight = clamp(unit.outputWeight, -7, 7);
  }

  return params;
}

function trainMultipleSteps(data, currentParams, hiddenUnitCount, steps, options) {
  let next = currentParams;
  for (let i = 0; i < steps; i++) {
    next = gradientStep(data, next, hiddenUnitCount, options);
  }
  return next;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function Icon({ type }) {
  if (type === "play") return <span aria-hidden="true">▶</span>;
  if (type === "pause") return <span aria-hidden="true">⏸</span>;
  if (type === "shuffle") return <span aria-hidden="true">↝</span>;
  if (type === "refresh") return <span aria-hidden="true">↻</span>;
  if (type === "reset") return <span aria-hidden="true">↺</span>;
  return null;
}

function CompactSlider({ label, value, min, max, step, onChange, disabled = false }) {
  const digits = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;

  return (
    <div className={`compact-slider ${disabled ? "is-disabled" : ""}`}>
      <div className="slider-header">
        <label>{label}</label>
        <span>{formatNumber(value, digits)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function HiddenUnitButtons({ value, onChange, disabled }) {
  return (
    <div className="control-block">
      <div className="block-label">中間層のユニット数</div>
      <div className="unit-count-buttons">
        {[1, 2, 3, 4, 5, 6].map((count) => (
          <button
            key={count}
            type="button"
            className={`mini-button ${value === count ? "active" : ""}`}
            disabled={disabled}
            onClick={() => onChange(count)}
          >
            {count}個
          </button>
        ))}
      </div>
    </div>
  );
}

function DatasetButtons({ value, onChange, disabled }) {
  return (
    <div className="control-block">
      <div className="block-label">データの複雑さ</div>
      <div className="dataset-buttons">
        {DATA_PATTERNS.map((pattern) => (
          <button
            key={pattern.id}
            type="button"
            className={`mini-button ${value === pattern.id ? "active" : ""}`}
            disabled={disabled}
            onClick={() => onChange(pattern.id)}
          >
            {pattern.shortLabel}
          </button>
        ))}
      </div>
    </div>
  );
}

function ParameterMiniBar({ value, min = -7, max = 7 }) {
  const zero = ((0 - min) / (max - min)) * 100;
  const pos = ((value - min) / (max - min)) * 100;
  const left = Math.min(zero, pos);
  const width = Math.abs(pos - zero);

  return (
    <div className="parameter-mini-bar">
      <div className="parameter-zero" style={{ left: `${zero}%` }} />
      <div className="parameter-value" style={{ left: `${left}%`, width: `${width}%` }} />
    </div>
  );
}

function UnitSummary({ unit, index, active, selected, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`unit-summary ${selected ? "selected" : ""} ${active ? "active-unit" : "inactive-unit"}`}
    >
      <div className="unit-title">
        <div>
          <span className="unit-dot" style={{ backgroundColor: UNIT_COLORS[index] }} />
          Unit {index + 1}
        </div>
        <span>{active ? "ON" : "OFF"}</span>
      </div>

      <div className="unit-values">
        <div>
          w<sub>in</sub>={formatNumber(unit.inputWeight, 1)}
          <ParameterMiniBar value={unit.inputWeight} />
        </div>
        <div>
          b<sub>h</sub>={formatNumber(unit.hiddenBias, 1)}
          <ParameterMiniBar value={unit.hiddenBias} />
        </div>
        <div>
          w<sub>out</sub>={formatNumber(unit.outputWeight, 1)}
          <ParameterMiniBar value={unit.outputWeight} />
        </div>
      </div>
    </button>
  );
}

export default function App() {
  const [hiddenUnitCount, setHiddenUnitCount] = useState(3);
  const [selectedUnitIndex, setSelectedUnitIndex] = useState(0);
  const [params, setParams] = useState(makeDefaultParams());
  const [sampleSize, setSampleSize] = useState(55);
  const [noise, setNoise] = useState(0.55);
  const [dataSeed, setDataSeed] = useState(1);
  const [datasetId, setDatasetId] = useState("hard");
  const [showTrueFunction, setShowTrueFunction] = useState(true);
  const [showUnitContributions, setShowUnitContributions] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSteps, setTrainingSteps] = useState(0);
  const [bestRmse, setBestRmse] = useState(null);
  const [learningRate, setLearningRate] = useState(0.022);
  const [stepsPerFrame, setStepsPerFrame] = useState(18);
  const trainingStartedRef = useRef(false);

  const width = 700;
  const height = 520;
  const padding = 50;
  const xMin = -3.2;
  const xMax = 3.2;

  const selectedPattern = DATA_PATTERNS.find((pattern) => pattern.id === datasetId) ?? DATA_PATTERNS[0];
  const yMin = selectedPattern.yMin;
  const yMax = selectedPattern.yMax;

  const xToSvg = (x) => padding + ((x - xMin) / (xMax - xMin)) * (width - padding * 2);
  const yToSvg = (y) => height - padding - ((y - yMin) / (yMax - yMin)) * (height - padding * 2);

  const data = useMemo(() => {
    const result = [];
    for (let i = 0; i < sampleSize; i++) {
      const r = pseudoRandom(dataSeed * 1000 + i);
      const x = xMin + r * (xMax - xMin);
      const yTrue = selectedPattern.fn(x);
      const y = yTrue + noise * selectedPattern.noiseScale * normalLikeNoise(dataSeed * 2000 + i);
      result.push({ x, y, yTrue });
    }
    return result;
  }, [sampleSize, noise, dataSeed, selectedPattern]);

  useEffect(() => {
    if (!isTraining) return undefined;

    let animationId;
    const animate = () => {
      setParams((prev) => {
        const trainable = trainingStartedRef.current
          ? prev
          : ensureTrainableParams(prev, hiddenUnitCount, dataSeed + trainingSteps + 1);
        trainingStartedRef.current = true;
        return trainMultipleSteps(data, trainable, hiddenUnitCount, stepsPerFrame, { learningRate });
      });
      setTrainingSteps((prev) => prev + stepsPerFrame);
      animationId = requestAnimationFrame(animate);
    };

    animationId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationId);
  }, [isTraining, data, hiddenUnitCount, learningRate, stepsPerFrame, dataSeed, trainingSteps]);

  const makeCurvePoints = (fn) => {
    const result = [];
    const steps = 280;
    for (let i = 0; i <= steps; i++) {
      const x = xMin + (i / steps) * (xMax - xMin);
      const y = fn(x);
      result.push(`${xToSvg(x)},${yToSvg(y)}`);
    }
    return result.join(" ");
  };

  const manualPoints = useMemo(
    () => makeCurvePoints((x) => predictMlp(params, x, hiddenUnitCount)),
    [params, hiddenUnitCount, datasetId]
  );

  const trueFunctionPoints = useMemo(() => makeCurvePoints(selectedPattern.fn), [selectedPattern]);

  const unitContributionPoints = useMemo(() => {
    return activeUnits(params, hiddenUnitCount).map((unit) =>
      makeCurvePoints((x) => params.outputBias + unitContribution(unit, x))
    );
  }, [params, hiddenUnitCount, datasetId]);

  const manualRmse = useMemo(() => calculateRmse(data, params, hiddenUnitCount), [data, params, hiddenUnitCount]);

  useEffect(() => {
    setBestRmse((prev) => (prev === null ? manualRmse : Math.min(prev, manualRmse)));
  }, [manualRmse]);

  const selectedUnit = params.units[selectedUnitIndex];
  const parameterCount = 3 * hiddenUnitCount + 1;

  const stopTraining = () => {
    setIsTraining(false);
  };

  const clearTrainingProgress = () => {
    setTrainingSteps(0);
    setBestRmse(null);
    trainingStartedRef.current = false;
  };

  const updateSelectedUnit = (key, value) => {
    stopTraining();
    clearTrainingProgress();
    setParams((prev) => {
      const next = cloneParams(prev);
      next.units[selectedUnitIndex][key] = value;
      return next;
    });
  };

  const updateOutputBias = (value) => {
    stopTraining();
    clearTrainingProgress();
    setParams((prev) => ({ ...prev, outputBias: value }));
  };

  const reset = () => {
    stopTraining();
    setHiddenUnitCount(3);
    setSelectedUnitIndex(0);
    setParams(makeDefaultParams());
    setSampleSize(55);
    setNoise(0.55);
    setDataSeed(1);
    setDatasetId("hard");
    setShowTrueFunction(true);
    setShowUnitContributions(true);
    setLearningRate(0.022);
    setStepsPerFrame(18);
    clearTrainingProgress();
  };

  const regenerateData = () => {
    stopTraining();
    setDataSeed((prev) => prev + 1);
    clearTrainingProgress();
  };

  const randomizeParams = () => {
    stopTraining();
    setParams((prev) => {
      const next = cloneParams(prev);
      next.outputBias = (pseudoRandom(dataSeed * 77 + 1) - 0.5) * 2;
      for (let j = 0; j < MAX_HIDDEN_UNITS; j++) {
        next.units[j] = {
          inputWeight: (pseudoRandom(dataSeed * 100 + j * 3 + 1) - 0.5) * 5.5,
          hiddenBias: (pseudoRandom(dataSeed * 100 + j * 3 + 2) - 0.5) * 5,
          outputWeight: (pseudoRandom(dataSeed * 100 + j * 3 + 3) - 0.5) * 5,
        };
      }
      return next;
    });
    clearTrainingProgress();
  };

  const toggleTraining = () => {
    setIsTraining((prev) => !prev);
  };

  const handleHiddenUnitCountChange = (count) => {
    stopTraining();
    setHiddenUnitCount(count);
    setSelectedUnitIndex((prev) => Math.min(prev, count - 1));
    clearTrainingProgress();
  };

  const handleDatasetChange = (id) => {
    stopTraining();
    const nextPattern = DATA_PATTERNS.find((pattern) => pattern.id === id) ?? DATA_PATTERNS[0];
    setDatasetId(id);
    setHiddenUnitCount(nextPattern.recommendedUnits);
    setSelectedUnitIndex(0);
    setNoise(Number((nextPattern.noiseScale * 1.0).toFixed(2)));
    clearTrainingProgress();
  };

  const gridLines = [];
  for (let x = -3; x <= 3; x += 1) {
    gridLines.push(
      <line
        key={`grid-x-${x}`}
        x1={xToSvg(x)}
        y1={yToSvg(yMin)}
        x2={xToSvg(x)}
        y2={yToSvg(yMax)}
        className="grid-line"
      />
    );
  }
  for (let y = Math.ceil(yMin / 2) * 2; y <= yMax; y += 2) {
    gridLines.push(
      <line
        key={`grid-y-${y}`}
        x1={xToSvg(xMin)}
        y1={yToSvg(y)}
        x2={xToSvg(xMax)}
        y2={yToSvg(y)}
        className="grid-line"
      />
    );
  }

  const tickLabels = [];
  for (let x = -3; x <= 3; x += 1) {
    if (x !== 0) {
      tickLabels.push(
        <text key={`x-label-${x}`} x={xToSvg(x)} y={yToSvg(0) + 18} textAnchor="middle" className="tick-label">
          {x}
        </text>
      );
    }
  }
  for (let y = Math.ceil(yMin / 2) * 2; y <= yMax; y += 2) {
    if (y !== 0) {
      tickLabels.push(
        <text key={`y-label-${y}`} x={xToSvg(0) - 12} y={yToSvg(y) + 4} textAnchor="end" className="tick-label">
          {y}
        </text>
      );
    }
  }

  const formula = `ŷ = ${formatNumber(params.outputBias, 2)} + Σ w_out,j tanh(w_in,j x + b_h,j)`;

  return (
    <div className="app">
      <div className="app-inner">
        <header className="app-header">
          <h1>3層MLPパラメータ・シミュレータ</h1>
          <div className="legend">
            <span><i className="legend-line mlp" />MLP</span>
            <span><i className="legend-line true" />真の関数</span>
            <span><i className="legend-dot data" />データ</span>
            <span><i className="legend-dot training" />探索中: {isTraining ? "ON" : "OFF"}</span>
          </div>
        </header>

        <main className="main-grid">
          <section className="left-column">
            <div className="card plot-card">
              <div className="plot-and-metrics">
                <svg viewBox={`0 0 ${width} ${height}`} className="plot-svg">
                  {gridLines}

                  <line x1={xToSvg(xMin)} y1={yToSvg(0)} x2={xToSvg(xMax)} y2={yToSvg(0)} className="axis-line" />
                  <line x1={xToSvg(0)} y1={yToSvg(yMin)} x2={xToSvg(0)} y2={yToSvg(yMax)} className="axis-line" />

                  {tickLabels}

                  <text x={xToSvg(xMax) + 16} y={yToSvg(0) + 5} className="axis-label">x</text>
                  <text x={xToSvg(0) - 5} y={yToSvg(yMax) - 16} className="axis-label">y</text>

                  {showTrueFunction && (
                    <polyline
                      points={trueFunctionPoints}
                      fill="none"
                      className="true-function-line"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}

                  {data.map((point, index) => (
                    <circle key={`data-${index}`} cx={xToSvg(point.x)} cy={yToSvg(point.y)} r="4.2" className="data-point" />
                  ))}

                  {showUnitContributions &&
                    unitContributionPoints.map((points, index) => (
                      <polyline
                        key={`unit-line-${index}`}
                        points={points}
                        fill="none"
                        stroke={UNIT_COLORS[index]}
                        strokeWidth={selectedUnitIndex === index ? 2.8 : 1.8}
                        strokeDasharray="5 6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity={selectedUnitIndex === index ? 0.75 : 0.35}
                      />
                    ))}

                  <polyline
                    points={manualPoints}
                    fill="none"
                    className="manual-line"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>

                <div className="metric-panel">
                  <div className="metric-box current">
                    <div className="metric-label">現在RMSE</div>
                    <div className="metric-value">{manualRmse.toFixed(3)}</div>
                  </div>
                  <div className="metric-box best">
                    <div className="metric-label">最良RMSE</div>
                    <div className="metric-value">{bestRmse === null ? "—" : bestRmse.toFixed(3)}</div>
                  </div>
                  <div className="metric-box steps">
                    <div className="metric-label">探索回数</div>
                    <div className="metric-value">{trainingSteps}</div>
                  </div>
                  <div className="formula-box">{formula}</div>

                  <div className="toggle-panel">
                    <label>
                      真の関数
                      <input type="checkbox" checked={showTrueFunction} onChange={(e) => setShowTrueFunction(e.target.checked)} />
                    </label>
                    <label>
                      ユニット別曲線
                      <input
                        type="checkbox"
                        checked={showUnitContributions}
                        onChange={(e) => setShowUnitContributions(e.target.checked)}
                      />
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div className="card data-card">
              <DatasetButtons value={datasetId} onChange={handleDatasetChange} disabled={isTraining} />
              <div className="data-sliders">
                <CompactSlider
                  label="サンプルサイズ"
                  value={sampleSize}
                  min={15}
                  max={180}
                  step={5}
                  disabled={isTraining}
                  onChange={(v) => {
                    stopTraining();
                    setSampleSize(v);
                    clearTrainingProgress();
                  }}
                />
                <CompactSlider
                  label="ノイズ"
                  value={noise}
                  min={0}
                  max={2.2}
                  step={0.05}
                  disabled={isTraining}
                  onChange={(v) => {
                    stopTraining();
                    setNoise(v);
                    clearTrainingProgress();
                  }}
                />
              </div>
              <button type="button" className="outline-button" onClick={regenerateData} disabled={isTraining}>
                <Icon type="refresh" /> データ再生成
              </button>
            </div>
          </section>

          <section className="right-column">
            <div className="card training-card">
              <div className="button-grid">
                <button type="button" className={`primary-button ${isTraining ? "stop" : ""}`} onClick={toggleTraining}>
                  <Icon type={isTraining ? "pause" : "play"} />{" "}
                  {isTraining ? "探索ストップ" : trainingSteps > 0 ? "探索再開" : "探索スタート"}
                </button>
                <button type="button" className="outline-button" onClick={randomizeParams} disabled={isTraining}>
                  <Icon type="shuffle" /> ランダム初期値
                </button>
              </div>
              <div className="training-sliders">
                <CompactSlider
                  label="学習率"
                  value={learningRate}
                  min={0.001}
                  max={0.08}
                  step={0.001}
                  onChange={setLearningRate}
                />
                <CompactSlider
                  label="探索速度"
                  value={stepsPerFrame}
                  min={1}
                  max={80}
                  step={1}
                  onChange={setStepsPerFrame}
                />
              </div>
            </div>

            <div className="card units-card">
              <div className="units-top">
                <HiddenUnitButtons value={hiddenUnitCount} onChange={handleHiddenUnitCountChange} disabled={isTraining} />
                <div className="parameter-count">
                  <div>パラメータ数</div>
                  <strong>{parameterCount}</strong>
                </div>
              </div>

              <div className="unit-grid">
                {params.units.map((unit, index) => (
                  <UnitSummary
                    key={index}
                    unit={unit}
                    index={index}
                    active={index < hiddenUnitCount}
                    selected={selectedUnitIndex === index}
                    onClick={() => setSelectedUnitIndex(Math.min(index, hiddenUnitCount - 1))}
                  />
                ))}
              </div>
            </div>

            <div className="card edit-card">
              <div className="edit-header">
                <strong>Unit {selectedUnitIndex + 1} を編集</strong>
                <span className="edit-dot" style={{ backgroundColor: UNIT_COLORS[selectedUnitIndex] }} />
              </div>

              <div className="edit-sliders">
                <CompactSlider
                  label="w_in"
                  value={selectedUnit.inputWeight}
                  min={-7}
                  max={7}
                  step={0.05}
                  disabled={isTraining}
                  onChange={(value) => updateSelectedUnit("inputWeight", value)}
                />
                <CompactSlider
                  label="b_h"
                  value={selectedUnit.hiddenBias}
                  min={-7}
                  max={7}
                  step={0.05}
                  disabled={isTraining}
                  onChange={(value) => updateSelectedUnit("hiddenBias", value)}
                />
                <CompactSlider
                  label="w_out"
                  value={selectedUnit.outputWeight}
                  min={-7}
                  max={7}
                  step={0.05}
                  disabled={isTraining}
                  onChange={(value) => updateSelectedUnit("outputWeight", value)}
                />
                <CompactSlider
                  label="b_out"
                  value={params.outputBias}
                  min={-6}
                  max={6}
                  step={0.05}
                  disabled={isTraining}
                  onChange={updateOutputBias}
                />
              </div>
            </div>

            <div className="reset-row">
              <button type="button" className="outline-button" onClick={reset}>
                <Icon type="reset" /> 初期値
              </button>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
