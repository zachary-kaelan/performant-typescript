import { Vector } from "./vector";

class Matrix {
    public numRows: number;
    public numCols: number;
    public count: number;
    public _compactedMatrix: Float64Array;

    // #region Getters and Setters
    public getAt(i: number, j: number): number {
        return this._compactedMatrix[i * this.numCols + j];
    }

    public setAt(i: number, j: number, value: number) {
        this._compactedMatrix[i * this.numCols + j] = value;
    }

    public getRow(i: number): Float64Array {
        let start = this.numCols * i;
        return this._compactedMatrix.slice(start, start + this.numCols);
    }

    public setRow(i: number, value: Float64Array) {
        if (value.length != this.numCols)
            throw new RangeError("Vector size needs to be the same as the number of columns in the matrix.");
        let start: number = i * this.numCols;
        this._compactedMatrix.set(value, i * this.numCols);
    }

    public getColumn(columnIndex: number): Vector {
        let column = new Vector(this.numRows);
        let num: number = columnIndex;
        for (let j: number = 0; j < this.numRows; j++) {
            column.setAt(j, this._compactedMatrix[num]);
            num += this.numCols;
        }
        return column;
    }

    public setColumn(columnIndex: number, column: Float64Array) {
        let index: number = columnIndex;
        for (let i: number = 0; i < this.numRows; i++) {
            this._compactedMatrix[index] = column[i];
            index += this.numCols;
        }
    }
    // #endregion

    // #region Initialization
    constructor(n: number);
    constructor(rows: number, cols: number);
    constructor(mat: Float64Array[]);
    constructor(compacted: Float64Array, cols: number)
    constructor(mat: Float64Array[], rowsCapacity: number, colsCapacity: number);
    constructor(compacted: Float64Array, cols: number, rowsCapacity: number, colsCapacity: number);
    constructor(...args: any[]) {
        if (args.length == 4) {
            let compacted = <Float64Array>args[0];
            let cols = <number>args[1];
            let rowsCapacity = <number>args[2];
            let colsCapacity = <number>args[3];

            this.numRows = rowsCapacity;
            this.numCols = colsCapacity;
            this.count = this.numRows * this.numCols;
            this._compactedMatrix = new Float64Array(this.count);

            let count: number = compacted.length;
            let rows: number = count % this.numCols;
            for (let i: number = 0; i < rows; i++) {
                for (let j: number = 0; j < cols; j++) {
                    this._compactedMatrix[i * this.numCols + j] = compacted[i * cols + j];
                }
            }
        }
        else if (args.length == 3) {
            let mat = <Float64Array[]>args[0];
            let rowsCapacity = <number>args[1];
            let colsCapacity = <number>args[2];

            this.numRows = rowsCapacity;
            this.numCols = colsCapacity;
            this.count = this.numRows * this.numCols;
            this._compactedMatrix = new Float64Array(this.count);

            let rows: number = mat.length;
            for (let i: number = 0; i < rows; i++) {
                var row = mat[i];
                let cols: number = row.length;
                for (let j: number = 0; j < cols; j++) {
                    this._compactedMatrix[i * this.numCols + j] = mat[i][j];
                }
            }
        }
        else if (args.length == 2) {
            if (args[0] instanceof Float64Array) {
                let compacted = <Float64Array>args[0];
                let cols = <number>args[1];

                this.numCols = cols;
                this.count = compacted.length;
                this.numRows = this.count / this.numCols;
                this._compactedMatrix = new Float64Array(compacted);
            }
            else {
                let rows = <number>args[0];
                let cols = <number>args[1];

                this.numRows = rows;
                this.numCols = cols;
                this.count = rows * cols;
                this._compactedMatrix = new Float64Array(this.count);
            }
        }
        else if (args.length == 1) {
            let arg = args[0];
            if (Array.isArray(arg)) {
                let mat = <Float64Array[]>arg;

                this.numRows = mat.length;
                this.numCols = mat[0].length;
                this.count = this.numRows * this.numCols;
                this._compactedMatrix = new Float64Array(this.count);
                for (let i: number = 0; i < this.numRows; i++) {
                    for (let j: number = 0; j < this.numCols; j++) {
                        this._compactedMatrix[i * this.numCols + j] = mat[i][j];
                    }
                }
            }
            else {
                let n = <number>args[0];
                this.numRows = n;
                this.numCols = n;
                this.count = n * n;
                this._compactedMatrix = new Float64Array(this.count);
            }
        }
    }

    public static identity(n: number): Matrix {
        let mat: Matrix = new Matrix(n);
        for (let i: number = 0; i < n; i++) {
            mat.setAt(i, i, 1.0);
        }
        return mat;
    }

    public static generateRandom(rows: number, cols: number): Matrix;
    public static generateRandom(rows: number, cols: number, minVal: number, maxVal: number): Matrix;
    public static generateRandom(rows: number, cols: number, generator: () => number): Matrix;
    public static generateRandom(...args: any[]): Matrix {
        let rows = <number>args[0];
        let cols = <number>args[1]

        if (args.length == 3) {
            let generator = <() => number>args[2];
            return new Matrix(rows, cols).elementWiseChange(e => generator());
        }

        let minVal = -1.0;
        let maxVal = 1.0;

        if (args.length == 4) {
            minVal = <number>args[2];
            maxVal = <number>args[3];
        }

        var range = maxVal - minVal;
        return new Matrix(rows, cols).elementWiseChange(
            n => range * Math.random() + minVal
        );
    }
    // #endregion

    // #region Helpers
    public elementWiseChange(change: (element: number) => number): Matrix
    {
        let mat = new Matrix(this.numRows, this.numCols);
        for (let e: number = 0; e < this.count; ++e) {
            mat._compactedMatrix[e] = change(this._compactedMatrix[e]);
        }
        return mat;
    }

    public elementWiseOp(other: Matrix, op: (element1: number, element2: number) => number): Matrix
    {
        if (this.numRows != other.numRows || this.numCols != other.numCols)
            throw new RangeError(`Matrix A is ${this.numRows}x${this.numCols} and Matrix B is ${other.numRows}x${other.numCols}, when they should be the same size.`);
        let mat = new Matrix(this.numRows, this.numCols);
        for (let e: number = 0; e < this.count; ++e) {
            mat._compactedMatrix[e] = op(this._compactedMatrix[e], other._compactedMatrix[e]);
        }
        return mat;
    }

    public getElement(index: number): number {
        return this._compactedMatrix[index];
    }

    public setElement(index: number, value: number) {
        this._compactedMatrix[index] = value;
    }
    // #endregion

    // #region Operators
    public add(num: number): Matrix;
    public add(other: Matrix): Matrix;
    public add(...args: any[]): Matrix {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e + arg);
        }
        else {
            let other = <Matrix>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 + e2);
        }
    }

    public subtract(num: number): Matrix;
    public subtract(other: Matrix): Matrix;
    public subtract(...args: any[]): Matrix {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e - arg);
        }
        else {
            let other = <Matrix>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 - e2);
        }
    }

    public multiply(num: number): Matrix;
    public multiply(other: Matrix): Matrix;
    public multiply(...args: any[]): Matrix {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e * arg);
        }
        else {
            let other = <Matrix>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 * e2);
        }
    }

    public divide(num: number): Matrix;
    public divide(other: Matrix): Matrix;
    public divide(...args: any[]): Matrix {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e / arg);
        }
        else {
            let other = <Matrix>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 / e2);
        }
    }

    public reciprocal(): Matrix {
        return this.elementWiseChange(e => 1 / e);
    }

    public negative(): Matrix {
        return this.elementWiseChange(e => -e);
    }

    public equals(other: Matrix): boolean;
    public equals(other: Matrix, epsilon: number): boolean; 
    public equals(...args: any[]): boolean {
        let other = <Matrix>args[0];
        let epsilon = 0;
        if (args.length == 2) {
            epsilon = <number>args[1];
        }

        if (other == null || other.numRows != this.numRows || other.numCols != this.numCols)
            return false;

        for (let e: number = 0; e < this.count; ++e) {
            if (Math.abs(this._compactedMatrix[e] - other._compactedMatrix[e]) > epsilon)
                return false;
        }

        return true;
    }
    // #endregion

    // #region Matrix Functions
    // #region Dot
    public tiledDot(other: Matrix): Matrix
    {
        if (this.numCols != other.numRows)
            throw new RangeError(`Matrix A has ${this.numCols} columns and Matrix B has ${other.numRows} rows, when they should be equal.`);
        let output = new Matrix(this.numRows, other.numCols);

        // fully associative cache consisting of M bytes
        const CACHE_SIZE = 400;
        // Pick a tile size T = sqrt(M)
        const TILE_SIZE = Math.sqrt(CACHE_SIZE);

        // for I from 1 to n in steps of T
        for (let I: number = 0; I < this.numRows; I += TILE_SIZE)
        {
            let tile_row_end = Math.min(I + TILE_SIZE, this.numRows);

            // for J from 1 to p in steps of T
            for (let J: number = 0; J < other.numCols; J += TILE_SIZE)
            {
                let tile_other_col_end = Math.min(J + TILE_SIZE, other.numCols);

                // for K from 1 to m in steps of T
                for (let K: number = 1; K < this.numCols; K += TILE_SIZE)
                {
                    let tile_col_end = Math.min(K + TILE_SIZE, this.numCols);

                    // for i from I to min(I + T, n)
                    for (let i: number = I; i < tile_row_end; i++)
                    {
                        let this_row = this.getRow(i);

                        // for j from J to min(J + T, p)
                        for (let j: number = J; j < tile_other_col_end; j++)
                        {
                            let sum = 0;

                            // for k from K to min(K + T, m)
                            for (let k: number = K; k < tile_col_end; k++) {
                                // set sum <- sum + A[i, k] * B[k, j]
                                sum += this_row[k] * other.getAt(k, j);
                            }

                            // set C[i, j] <- C[i, j] + sum
                            let index = i * output.numCols + j;
                            output._compactedMatrix[index] += sum;
                        }
                    }
                }
            }
        }

        return output;
    }

    public dot(other: Matrix): Matrix {
        if (this.numCols != other.numRows)
            throw new Error(`Matrix A has ${this.numCols} columns and Matrix B has ${other.numRows} rows, when they should be equal.`);

        let output = new Matrix(this.numRows, other.numCols);
        let mat1Index: number = 0;

        for (let i: number = 0; i < this.numRows; i++) {
            for (let j: number = 0; j < other.numCols; j++) {
                let sum: number = 0;
                for (let k: number = 0; k < this.numCols; k++) {
                    sum += this._compactedMatrix[mat1Index + k] * other.getAt(k, j);
                }
                output.setAt(i, j, sum);
            }

            mat1Index += this.numCols;
        }

        return output;
    }

    public dotVec(vector: Vector): Vector {
        if (this.numCols != vector.count)
            throw new RangeError(`Matrix has ${this.numCols} columns, and Vector is size ${vector.count}. The two need to be equal.`);

        let vectorProduct: Float64Array = new Float64Array(vector.count);
        for (let e: number = 0; e < this.count; ++e) {
            let index: number = e % vector.count;
            vectorProduct[index] += this._compactedMatrix[e] * vector.getAt(index);
        }

        return new Vector(vectorProduct);
    }
    // #endregion

    public checkSymmetry(): boolean {
        let symmetrical = true;
        let rowStart: number = 0;
        for (let i: number = 0; symmetrical && (i < this.numRows); i++) {
            for (let j: number = 0; symmetrical && (j < this.numCols); j++) {
                symmetrical = (this._compactedMatrix[rowStart + j] == this.getAt(j, i));
            }
        }
        return symmetrical;
    }

    public T(): Matrix {
        let output = new Matrix(this.numCols, this.numRows);
        for (let i: number = 0; i < this.numRows; i++) {
            for (let j: number = 0; j < this.numCols; j++) {
                output.setAt(j, i, this.getAt(i, j));
            }
        }

        return output;
    }

    public decompose(perm: Int32Array): [Matrix, number] {
        // no idea if this works, or how it works

        if (this.numRows != this.numCols)
            throw new Error("Trying to decompose a non-square matrix.");

        let output = this.duplicate();
        var permTemp = new Int32Array(this.numRows);

        for (let i: number = 0; i < this.numRows; i++) {
            permTemp[i] = i;
        }

        let toggleTemp: number = 1;
        // toggle tracks row swaps
        // +1 - greater-than even
        // -1 - greater-than odd

        let rowSwap = (row: number, col: number) =>
        {
            let rowPtr: Float64Array = output[row];
            output[row] = output[col];
            output[col] = rowPtr;

            // swap perm info
            let tmp: number = permTemp[row];
            permTemp[row] = permTemp[col];
            permTemp[row] = tmp;

            // adjust the row-swap toggle
            toggleTemp = -toggleTemp;
        }

        for (let j: number = 0; j < this.numRows - 1; j++) {
            let colMax: number = Math.abs(output.getAt(j, j));
            let pRow: number = j;

            for (let i: number = j + 1; i < this.numRows; i++) {
                let num: number = Math.abs(output.getAt(i, j));
                if (num > colMax) {
                    colMax = num;
                    pRow = i;
                }
            }

            // if largest value not on pivot, swap rows
            if (pRow != j)
                rowSwap(pRow, j);

            if (output.getAt(j, j) == 0.0) {
                // find a good row to swap
                let goodRow: number = -1;
                for (let i: number = j + 1; i < this.numRows; i++) {
                    if (output.getAt(i, j) != 0.0)
                        goodRow = i;
                }

                if (goodRow == -1)
                    throw new EvalError("No good row, can't use Doolittle's method.");

                // swap rows so 0.0 no longer on diagonal
                rowSwap(goodRow, j);
            }

            for (let i: number = j + 1; i < this.numRows; i++) {
                output.setAt(i, j, output.getAt(i, j) / output.getAt(j, j));
                for (let k: number = j + 1; k < this.numRows; k++) {
                    output.setAt(i, k, output.getAt(i, k) - output.getAt(i, j) * output.getAt(j, k));
                }
            }
        }

        perm = permTemp;
        return [output, toggleTemp];
    }

    public invert(): Matrix {
        if (this.numRows != this.numCols)
            throw new Error("Trying to invert a non-square matrix.");

        // Cramer's rule
        if (this.numRows == 2)
            return new Matrix(
                new Array<Float64Array>(
                    new Float64Array(
                        [this._compactedMatrix[3],
                        -this._compactedMatrix[1]]
                    ), new Float64Array(
                        [-this._compactedMatrix[2],
                        this._compactedMatrix[0]]
                    )
                )
            ).multiply(1.0 / this.determinant());
        else if (this.numRows == 3) {
            let a: number = this._compactedMatrix[0];
            let b: number = this._compactedMatrix[1];
            let c: number = this._compactedMatrix[2];
            let d: number = this._compactedMatrix[3];
            let e: number = this._compactedMatrix[4];
            let f: number = this._compactedMatrix[5];
            let g: number = this._compactedMatrix[6];
            let h: number = this._compactedMatrix[7];
            let i: number = this._compactedMatrix[8];
            let A: number = (e * i) - (f * h);
            let B: number = -1 * ((d * i) - (f * g));
            let C: number = (d * h) - (e * g);

            // rule of Sarrus
            let determinant: number = (a * A) + (b * B) + (c * C);

            return new Matrix(
                new Array<Float64Array>(
                    new Float64Array(
                        [
                            A,
                            -((b * i) - (c * h)),
                            (b * f) - (c * e)
                        ]
                    ), new Float64Array(
                        [
                            B,
                            (a * i) - (c * g),
                            -((a * f) - (c * d))
                        ]
                    ), new Float64Array(
                        [
                            C,
                            -((a * h) - (b * g)),
                            (a * e) - (b * d)
                        ]
                    )
                )
            ).multiply(1.0 / determinant);

        }
                    
        let lum: Matrix = null;
        let perm: Int32Array = null;
        let toggle: number = 1;

        try {
            [lum, toggle] = this.decompose(perm);
        }
        catch
        {
            throw new Error("Matrix cannot be inverted using supported methods.");
        }

        let output = this.duplicate();
        let bTemp: Float64Array = new Float64Array(this.numRows);
        for (let i: number = 0; i < this.numRows; i++) {
            for (let j: number = 0; j < this.numCols; j++) {
                if (i == perm[j])
                    bTemp[j] = 1.0;
                else
                    bTemp[j] = 0.0;
            }

            let x: Float64Array = lum.helperSolve(bTemp);

            for (let j: number = 0; j < this.numRows; j++) {
                output.setAt(j, i, x[j]);
            }
        }

        return output;
    }

    public determinant(): number {
        if (this.numRows != this.numCols)
            throw new EvalError("Cannot get determinant of non-square matrix.");
        if (this.numRows == 2)
            return (this._compactedMatrix[0] * this._compactedMatrix[3]) -
                (this._compactedMatrix[1] * this._compactedMatrix[2]);
        else if (this.numRows == 3)
            return (this._compactedMatrix[0] * this._compactedMatrix[4] * this._compactedMatrix[8]) +
                (this._compactedMatrix[1] * this._compactedMatrix[5] * this._compactedMatrix[6]) +
                (this._compactedMatrix[2] * this._compactedMatrix[3] * this._compactedMatrix[7]) -
                (this._compactedMatrix[2] * this._compactedMatrix[4] * this._compactedMatrix[6]) -
                (this._compactedMatrix[1] * this._compactedMatrix[3] * this._compactedMatrix[8]) -
                (this._compactedMatrix[0] * this._compactedMatrix[5] * this._compactedMatrix[7]);

        let perm: Int32Array = null;
        let lum: Matrix = null;
        let toggle: number = 1;
        
        [lum, toggle] = this.decompose(perm);
        if (lum == null)
            throw new Error("Unable to compute determinant");
        let result: number = toggle;
        for (let i: number = 0; i < lum.numRows; i++) {
            result *= lum.getAt(i, i);
        }
        return result;
    }

    private helperSolve(b: Float64Array): Float64Array {
        // before calling, permute b using the perm array
        let output: Float64Array = new Float64Array(this.numRows);
        output.set(b);

        for (let i: number = 1; i < this.numRows; i++) {
            let sum: number = output[i];
            for (let j: number = 0; j < i; j++) {
                sum -= this.getAt(i, j) * output[j];
            }
            output[i] = sum / this.getAt(i, i);
        }

        return output;
    }

    private systemSolve(b: Float64Array): Float64Array {
        // 1. decompose A
        let perm: Int32Array = null;
        let lum: Matrix = null;
        let toggle: number = 1;
        
        [lum, toggle] = this.decompose(perm);
        if (lum == null)
            return null;

        // 2. permute b according to perm[] into bp
        let bp: Float64Array = new Float64Array(b.length);
        for (let i: number = 0; i < this.numRows; i++) {
            bp[i] = b[perm[i]];
        }

        // 3. call helper
        return lum.helperSolve(bp);
    }
    // #endregion

    // #region StatisticsFunctions
    public getColumnMeans(): Float64Array {
        let sums: Float64Array = new Float64Array(this.numCols);
        let rowStart: number = 0;
        for (let i: number = 0; i < this.numRows; i++) {
            for (let j: number = 0; j < this.numCols; j++) {
                sums[j] += this._compactedMatrix[rowStart + j];
            }
            rowStart += this.numCols;
        }
        return sums.map(s => s / this.numRows);
    }

    public getCovarianceMatrix(): Matrix {
        let covarMat = new Matrix(this.numCols);
        let vectors = [...Array(this.numCols).keys()].map(j => this.getColumn(j));
        for (let j: number = 0; j < this.numCols - 1; j++) {
            let vec = this.getColumn(j);
            covarMat.setAt(j, j, vec.variance());
        for (let k: number = j + 1; k < this.numCols; k++) {
            let covariance: number = vec.elementWiseAggregate(vectors[k], (sum, jVal, kVal) => sum + (jVal * kVal)) / this.count;
            covarMat.setAt(j, k, covariance);
            covarMat.setAt(k, j, covariance);
        }
    }
    covarMat._compactedMatrix[covarMat.count - 1] = vectors[vectors.length - 1].variance();
    return covarMat;
}

    // Skipped the Principle Component Analysis function because converting that also requires converting the 1182 line 'EigenvalueDecomposition' class

    // #endregion

    // #region Resize
    public resize(newRowCapacity: number);
    public resize(newRowCapacity: number, newColCapacity: number);

    public resize(...args: any[]) {
        let newRowCapacity = <number>args[0];
        let newColCapacity = this.numCols;
        let newCount = newRowCapacity * newColCapacity;
        if (args.length == 1) {
            newColCapacity = <number>args[1];
        }

        let rowsChange = newRowCapacity - this.numRows;
        let colsChange = newColCapacity - this.numCols;
        if (rowsChange == 0 && colsChange == 0) {
            return;
        }
        if (rowsChange < 0 || colsChange < 0) {
            throw new RangeError("New matrix size cannot be smaller than the old one.");
        }

        let newCompacted = new Float64Array(newCount);

        if (colsChange == 0) {
            newCompacted.set(this._compactedMatrix, 0);
        }
        else {
            let srcIndex = 0;
            let destIndex = 0;
            for (let i = 0; i < this.numRows; i++) {
                newCompacted.set(this._compactedMatrix.slice(srcIndex, srcIndex + this.numCols), destIndex);
                srcIndex += this.numCols;
                destIndex += newColCapacity;
            }
        }

        this.numRows = newRowCapacity;
        this.numCols = newColCapacity;
        this.count = newCount;
        this._compactedMatrix = newCompacted;
    }
    // #endregion

    // #region Extending
    public addRow(row: Float64Array)
    {
        if (row.length != this.numCols) {
            throw new RangeError('Tried to add a row to a matrix when it had more columns.');
        }

        let newCompacted = new Float64Array(this.count + this.numCols);
        newCompacted.set(this._compactedMatrix);
        newCompacted.set(row, this._compactedMatrix.length);
        this._compactedMatrix = newCompacted;
        ++this.numRows;
        this.count += this.numCols;
    }

    public addColumn(column: Float64Array) {
        if (column.length != this.numRows) {
            throw new RangeError("Cannot add a column to a matrix unless it fits the matrix's number of rows.");
        }

        let destIndex = this.numCols;
        this.resize(this.numCols, this.numCols + 1);
        for (let i = 0; i < this.numRows; i++) {
            this._compactedMatrix[destIndex] = column[i];
            destIndex += this.numCols;
        }
    }

    public glueWith(other: Matrix, vertical?: boolean): Matrix
    {
        if (other.numCols == this.numCols && vertical) {
            let newCompacted = new Float64Array(this.count + other.count);
            newCompacted.set(this._compactedMatrix);
            newCompacted.set(other._compactedMatrix, this.count);
            return new Matrix(newCompacted, this.numRows + other.numRows);
        }
        else if (other.numRows == this.numRows && !vertical) {
            let output = this.duplicate();
            output.resize(this.numRows, this.numCols + other.numCols);

            let srcIndex = 0;
            let destIndex = this.numCols;
            for (let i = 0; i < this.numRows; i++) {
                output._compactedMatrix.set(other._compactedMatrix.slice(srcIndex, srcIndex + other.numCols), destIndex);
                srcIndex += other.numCols;
                destIndex += output.numCols;
            }

            return output;
        }
        else {
            throw new RangeError("Matrices need compatible sizes on the specified side to be glued.");
        }
    }
    // #endregion

    public duplicate(): Matrix {
        return new Matrix(new Float64Array(this._compactedMatrix), this.numCols);
    }

    public toArray(): Float64Array[] {
        let output: Float64Array[] = new Array<Float64Array>(this.numRows);
        for (let i: number = 0; i < this.numRows; i++) {
            let row = new Float64Array(this.numCols);
            for (let j: number = 0; j < this.numCols; j++) {
                row[j] = this.getAt(i, j);
            }
            output[i] = row;
        }
        return output;
    }

    public toString(): string {
        let str = "[[";
        for (let i = 0; i < this.numRows; i++) {
            if (i > 0) {
                str += "\n ";
            }
            let row = this.getRow(i);
            str += row.join(' ');
            str += "]";
        }
        str += "]";
        return str;
    }
}

export { Matrix };
