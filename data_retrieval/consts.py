EVALSCRIPT_CLOUDLESS = """
//VERSION=3
function setup() {
    return {
        input: ["B08", "B04", "B03", "B02", "SCL"],
        output: {
            bands: 4,
            sampleType: "INT16"
        },
        mosaicking: "ORBIT"
    }
}

function getFirstQuartileValue(values) {
    values.sort((a,b) => a-b);
    return getFirstQuartile(values);
}

function getFirstQuartile(sortedValues) {
    var index = Math.floor(sortedValues.length / 4);
    return sortedValues[index];
}

function validate(sample) {
    // Define codes as invalid:
    const invalid = [
        0, // NO_DATA
        1, // SATURATED_DEFECTIVE
        3, // CLOUD_SHADOW
        7, // CLOUD_LOW_PROBA
        8, // CLOUD_MEDIUM_PROBA
        9, // CLOUD_HIGH_PROBA
        10 // THIN_CIRRUS
    ]
    return !invalid.includes(sample.SCL)
}

function evaluatePixel(samples) {
    var valid = samples.filter(validate);
    if (valid.length > 0 ) {
        let cloudless = {
            b08: getFirstQuartileValue(valid.map(s => s.B08)),
            b04: getFirstQuartileValue(valid.map(s => s.B04)),
            b03: getFirstQuartileValue(valid.map(s => s.B03)),
            b02: getFirstQuartileValue(valid.map(s => s.B02)),
        }
        let ndvi = ((cloudless.b08 - cloudless.b04) / (cloudless.b08 + cloudless.b04))
        // This applies a scale factor so the data can be saved as an int
        let scale = [cloudless.b04, cloudless.b03, cloudless.b02, ndvi].map(v => v*10000);
        return scale
    }
    // If there isn't enough data, return NODATA
    return [-32768, -32768, -32768, -32768]
}
"""
