TRUE_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["red", "green", "blue", "clear", "dataMask"],
    output: { bands: 4, sampleType: "UINT8" },
    mosaicking: "ORBIT"
  };
}

function evaluatePixel(samples) {
  let red = 0;
  let green = 0;
  let blue = 0;
  let count = 0;

  for (let i = 0; i < samples.length; i++) {
    const valid = samples[i].dataMask * samples[i].clear;
    if (valid !== 0) {
      red += samples[i].red;
      green += samples[i].green;
      blue += samples[i].blue;
      count += 1;
    }
  }

  if (count === 0) {
    return [0, 0, 0, 0];
  }

  return [
    Math.min(255, Math.max(0, Math.round((red / count) / 4))),
    Math.min(255, Math.max(0, Math.round((green / count) / 4))),
    Math.min(255, Math.max(0, Math.round((blue / count) / 4))),
    255
  ];
}
"""

FALSE_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["nir", "red", "green", "clear", "dataMask"],
    output: { bands: 4, sampleType: "UINT8" },
    mosaicking: "ORBIT"
  };
}

function evaluatePixel(samples) {
  let nir = 0;
  let red = 0;
  let green = 0;
  let count = 0;

  for (let i = 0; i < samples.length; i++) {
    const valid = samples[i].dataMask * samples[i].clear;
    if (valid !== 0) {
      nir += samples[i].nir;
      red += samples[i].red;
      green += samples[i].green;
      count += 1;
    }
  }

  if (count === 0) {
    return [0, 0, 0, 0];
  }

  return [
    Math.min(255, Math.max(0, Math.round((nir / count) / 4))),
    Math.min(255, Math.max(0, Math.round((red / count) / 4))),
    Math.min(255, Math.max(0, Math.round((green / count) / 4))),
    255
  ];
}
"""

MEAN_NDWI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["green", "nir", "clear", "dataMask"],
    output: { bands: 1, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}

function evaluatePixel(samples) {
  let sumNdwi = 0;
  let count = 0;

  for (let i = 0; i < samples.length; i++) {
    const valid = samples[i].dataMask * samples[i].clear;
    const denom = samples[i].green + samples[i].nir;

    if (valid !== 0 && denom !== 0) {
      sumNdwi += (samples[i].green - samples[i].nir) / denom;
      count += 1;
    }
  }

  if (count === 0) {
    return [NaN];
  }

  return [sumNdwi / count];
}
"""
