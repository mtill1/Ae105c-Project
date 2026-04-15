"""Lambert solver (Izzo + Lancaster-Blanchard-Gooding fallback).

Ported from MATLAB code by Rody P.S. Oldenhuis (2-clause BSD).
Original Izzo algorithm by Dr. D. Izzo (ESA/ACT).
"""

import numpy as np

# Precomputed coefficients for the sigmax series (25 terms).
_AN = np.array([
    4.000000000000000e-001, 2.142857142857143e-001, 4.629629629629630e-002,
    6.628787878787879e-003, 7.211538461538461e-004, 6.365740740740740e-005,
    4.741479925303455e-006, 3.059406328320802e-007, 1.742836409255060e-008,
    8.892477331109578e-010, 4.110111531986532e-011, 1.736709384841458e-012,
    6.759767240041426e-014, 2.439123386614026e-015, 8.203411614538007e-017,
    2.583771576869575e-018, 7.652331327976716e-020, 2.138860629743989e-021,
    5.659959451165552e-023, 1.422104833817366e-024, 3.401398483272306e-026,
    7.762544304774155e-028, 1.693916882090479e-029, 3.541295006766860e-031,
    7.105336187804402e-033,
])


def lambert(r1vec, r2vec, tf, m, muC):
    """Solve Lambert's problem (Izzo, with Lancaster-Blanchard fallback).

    Parameters
    ----------
    r1vec, r2vec : array-like, shape (3,)
        Position vectors [km].
    tf : float
        Time of flight [days]. Negative for long-way.
    m : int
        Number of complete revolutions. Negative for left branch.
    muC : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    V1, V2 : ndarray, shape (3,)
        Terminal velocity vectors [km/s].
    extremal_distances : ndarray, shape (2,)
        [min_distance, max_distance] to central body [km].
    exitflag : int
        +1 success, -1 no solution, -2 both solvers failed.
    """
    r1vec = np.asarray(r1vec, dtype=float)
    r2vec = np.asarray(r2vec, dtype=float)

    tol = 1e-14
    bad = False
    days = 86400.0

    # Work with non-dimensional units
    r1 = np.sqrt(np.dot(r1vec, r1vec))
    r1vec = r1vec / r1
    V = np.sqrt(muC / r1)
    r2vec = r2vec / r1
    T = r1 / V
    tf = tf * days / T  # transform to non-dimensional time

    # Geometry parameters (non-dimensional)
    mr2vec = np.sqrt(np.dot(r2vec, r2vec))
    dth = np.arccos(max(-1.0, min(1.0, np.dot(r1vec, r2vec) / mr2vec)))

    # Branch selection
    leftbranch = np.sign(m) if m != 0 else 1
    longway = np.sign(tf) if tf != 0 else 1
    m = abs(m)
    tf = abs(tf)
    if longway < 0:
        dth = 2 * np.pi - dth

    # Derived quantities
    c = np.sqrt(1 + mr2vec ** 2 - 2 * mr2vec * np.cos(dth))
    s = (1 + mr2vec + c) / 2
    a_min = s / 2
    Lambda = np.sqrt(mr2vec) * np.cos(dth / 2) / s

    crossprd = np.cross(r1vec, r2vec)
    mcr = np.linalg.norm(crossprd)
    nrmunit = crossprd / mcr

    # Initial values
    logt = np.log(tf)

    if m == 0:
        inn1 = -0.5233
        inn2 = +0.5233
        x1 = np.log(1 + inn1)
        x2 = np.log(1 + inn2)
    else:
        if leftbranch < 0:
            inn1 = -0.5234
            inn2 = -0.2234
        else:
            inn1 = +0.7234
            inn2 = +0.5234
        x1 = np.tan(inn1 * np.pi / 2)
        x2 = np.tan(inn2 * np.pi / 2)

    # Initial estimate is always ellipse
    xx = np.array([inn1, inn2])
    aa = a_min / (1 - xx ** 2)
    bbeta = longway * 2 * np.arcsin(np.sqrt((s - c) / 2.0 / aa))
    aalfa = 2 * np.arccos(np.clip(xx, -1, 1))

    # Time of flight via Lagrange expression
    y12 = aa * np.sqrt(aa) * ((aalfa - np.sin(aalfa)) - (bbeta - np.sin(bbeta)) + 2 * np.pi * m)

    if m == 0:
        y1 = np.log(y12[0]) - logt
        y2 = np.log(y12[1]) - logt
    else:
        y1 = y12[0] - tf
        y2 = y12[1] - tf

    # Newton-Raphson iterations
    err = np.inf
    iterations = 0
    xnew = 0.0
    while err > tol:
        iterations += 1
        xnew = (x1 * y2 - y1 * x2) / (y2 - y1)

        if m == 0:
            x = np.exp(xnew) - 1
        else:
            x = np.arctan(xnew) * 2 / np.pi

        a = a_min / (1 - x ** 2)
        if x < 1:  # ellipse
            beta = longway * 2 * np.arcsin(np.sqrt((s - c) / 2.0 / a))
            alfa = 2 * np.arccos(max(-1.0, min(1.0, x)))
        else:  # hyperbola
            alfa = 2 * np.arccosh(x)
            beta = longway * 2 * np.arcsinh(np.sqrt((s - c) / (-2.0 * a)))

        if a > 0:
            tof = a * np.sqrt(a) * ((alfa - np.sin(alfa)) - (beta - np.sin(beta)) + 2 * np.pi * m)
        else:
            tof = -a * np.sqrt(-a) * ((np.sinh(alfa) - alfa) - (np.sinh(beta) - beta))

        if m == 0:
            ynew = np.log(tof) - logt
        else:
            ynew = tof - tf

        x1 = x2
        x2 = xnew
        y1 = y2
        y2 = ynew
        err = abs(x1 - xnew)

        if iterations > 15:
            bad = True
            break

    # If Newton-Raphson failed, use Lancaster-Blanchard fallback
    if bad:
        return lambert_lancaster_blanchard(
            r1vec * r1, r2vec * r1, longway * tf * T, leftbranch * m, muC
        )

    # Convert converged x
    if m == 0:
        x = np.exp(xnew) - 1
    else:
        x = np.arctan(xnew) * 2 / np.pi

    # Semi-major axis
    a = a_min / (1 - x ** 2)

    # Calculate psi
    if x < 1:  # ellipse
        beta = longway * 2 * np.arcsin(np.sqrt((s - c) / 2.0 / a))
        alfa = 2 * np.arccos(max(-1.0, min(1.0, x)))
        psi = (alfa - beta) / 2
        eta2 = 2 * a * np.sin(psi) ** 2 / s
        eta = np.sqrt(eta2)
    else:  # hyperbola
        beta = longway * 2 * np.arcsinh(np.sqrt((c - s) / 2.0 / a))
        alfa = 2 * np.arccosh(x)
        psi = (alfa - beta) / 2
        eta2 = -2 * a * np.sinh(psi) ** 2 / s
        eta = np.sqrt(eta2)

    # Unit normal
    ih = longway * nrmunit

    # Unit vector for normalized r2vec
    r2n = r2vec / mr2vec

    # Cross products
    crsprd1 = np.cross(ih, r1vec)
    crsprd2 = np.cross(ih, r2n)

    # Radial and tangential directions for departure velocity
    Vr1 = 1 / eta / np.sqrt(a_min) * (2 * Lambda * a_min - Lambda - x * eta)
    Vt1 = np.sqrt(mr2vec / a_min / eta2 * np.sin(dth / 2) ** 2)

    # Radial and tangential directions for arrival velocity
    Vt2 = Vt1 / mr2vec
    Vr2 = (Vt1 - Vt2) / np.tan(dth / 2) - Vr1

    # Terminal velocities
    V1_out = (Vr1 * r1vec + Vt1 * crsprd1) * V
    V2_out = (Vr2 * r2n + Vt2 * crsprd2) * V

    exitflag = 1

    # Minimum/maximum distances (use un-transformed vectors)
    extremal_distances = minmax_distances(
        r1vec * r1, r1, r2vec * r1, mr2vec * r1, dth, a * r1, V1_out, V2_out, m, muC
    )

    return V1_out, V2_out, extremal_distances, exitflag


def lambert_lancaster_blanchard(r1vec, r2vec, tf, m, muC):
    """Lancaster-Blanchard solver with Gooding improvements (fallback)."""
    r1vec = np.asarray(r1vec, dtype=float)
    r2vec = np.asarray(r2vec, dtype=float)

    tol = 1e-12

    r1 = np.sqrt(np.dot(r1vec, r1vec))
    r2 = np.sqrt(np.dot(r2vec, r2vec))
    r1unit = r1vec / r1
    r2unit = r2vec / r2
    crsprod = np.cross(r1vec, r2vec)
    mcrsprd = np.linalg.norm(crsprod)
    th1unit = np.cross(crsprod / mcrsprd, r1unit)
    th2unit = np.cross(crsprod / mcrsprd, r2unit)

    dth = np.arccos(max(-1.0, min(1.0, np.dot(r1vec, r2vec) / r1 / r2)))

    # Long-way
    longway = np.sign(tf) if tf != 0 else 1
    tf = abs(tf)
    if longway < 0:
        dth = dth - 2 * np.pi

    # Left branch
    leftbranch = np.sign(m) if m != 0 else 1
    m = abs(m)

    # Constants
    c = np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(dth))
    s = (r1 + r2 + c) / 2
    T = np.sqrt(8 * muC / s ** 3) * tf
    q = np.sqrt(r1 * r2) / s * np.cos(dth / 2)

    # Initial values (Gooding)
    T0 = lancaster_blanchard(0, q, m)[0]
    Td = T0 - T
    phr = np.mod(2 * np.arctan2(1 - q ** 2, 2 * q), 2 * np.pi)

    # Pessimistic initial output
    V1 = np.full(3, np.nan)
    V2 = np.full(3, np.nan)
    extremal_distances = np.array([np.nan, np.nan])

    # Single-revolution case
    if m == 0:
        x01 = T0 * Td / 4 / T
        if Td > 0:
            x0 = x01
        else:
            x01 = Td / (4 - Td)
            x02 = -np.sqrt(-Td / (T + T0 / 2))
            W = x01 + 1.7 * np.sqrt(2 - phr / np.pi)
            if W >= 0:
                x03 = x01
            else:
                x03 = x01 + (-W) ** (1.0 / 16) * (x02 - x01)
            lam = 1 + x03 * (1 + x01) / 2 - 0.03 * x03 ** 2 * np.sqrt(1 + x01)
            x0 = lam * x03

        if x0 < -1:
            exitflag = -1
            return V1, V2, extremal_distances, exitflag

    # Multi-revolution case
    else:
        xMpi = 4 / (3 * np.pi * (2 * m + 1))
        if phr < np.pi:
            xM0 = xMpi * (phr / np.pi) ** (1.0 / 8)
        elif phr > np.pi:
            xM0 = xMpi * (2 - (2 - phr / np.pi) ** (1.0 / 8))
        else:
            xM0 = 0.0

        # Halley's method for minimum Tp(x)
        xM = xM0
        Tp = np.inf
        iterations = 0
        while abs(Tp) > tol:
            iterations += 1
            _, Tp, Tpp, Tppp = lancaster_blanchard(xM, q, m)
            xMp = xM
            xM = xM - 2 * Tp * Tpp / (2 * Tpp ** 2 - Tp * Tppp)
            if iterations % 7:
                xM = (xMp + xM) / 2
            if iterations > 25:
                exitflag = -2
                return V1, V2, extremal_distances, exitflag

        if xM < -1 or xM > 1:
            exitflag = -1
            return V1, V2, extremal_distances, exitflag

        TM = lancaster_blanchard(xM, q, m)[0]

        if TM > T:
            exitflag = -1
            return V1, V2, extremal_distances, exitflag

        # Two initial values for second solution
        TmTM = T - TM
        T0mTM = T0 - TM
        _, Tp, Tpp, _ = lancaster_blanchard(xM, q, m)

        if leftbranch > 0:
            x = np.sqrt(TmTM / (Tpp / 2 + TmTM / (1 - xM) ** 2))
            W = xM + x
            W = 4 * W / (4 + TmTM) + (1 - W) ** 2
            x0 = x * (1 - (1 + m + (dth - 1 / 2)) /
                       (1 + 0.15 * m) * x * (W / 2 + 0.03 * x * np.sqrt(W))) + xM
            if x0 > 1:
                exitflag = -1
                return V1, V2, extremal_distances, exitflag
        else:
            if Td > 0:
                x0 = xM - np.sqrt(TM / (Tpp / 2 - TmTM * (Tpp / 2 / T0mTM - 1 / xM ** 2)))
            else:
                x00 = Td / (4 - Td)
                W = x00 + 1.7 * np.sqrt(2 * (1 - phr))
                if W >= 0:
                    x03 = x00
                else:
                    x03 = x00 - np.sqrt((-W) ** (1.0 / 8)) * (x00 + np.sqrt(-Td / (1.5 * T0 - Td)))
                W = 4 / (4 - Td)
                lam = (1 + (1 + m + 0.24 * (dth - 1 / 2)) /
                        (1 + 0.15 * m) * x03 * (W / 2 - 0.03 * x03 * np.sqrt(W)))
                x0 = x03 * lam

            if x0 < -1:
                exitflag = -1
                return V1, V2, extremal_distances, exitflag

    # Halley's method to find root
    x = x0
    Tx = np.inf
    iterations = 0
    while abs(Tx) > tol:
        iterations += 1
        Tx, Tp, Tpp, _ = lancaster_blanchard(x, q, m)
        Tx = Tx - T
        xp = x
        x = x - 2 * Tx * Tp / (2 * Tp ** 2 - Tx * Tpp)
        if iterations % 7:
            x = (xp + x) / 2
        if iterations > 25:
            exitflag = -2
            return V1, V2, extremal_distances, exitflag

    # Terminal velocities
    gamma = np.sqrt(muC * s / 2)
    if c == 0:
        sigma = 1.0
        rho = 0.0
        z = abs(x)
    else:
        sigma = 2 * np.sqrt(r1 * r2 / (c ** 2)) * np.sin(dth / 2)
        rho = (r1 - r2) / c
        z = np.sqrt(1 + q ** 2 * (x ** 2 - 1))

    # Radial component
    Vr1 = +gamma * ((q * z - x) - rho * (q * z + x)) / r1
    Vr1vec = Vr1 * r1unit
    Vr2 = -gamma * ((q * z - x) + rho * (q * z + x)) / r2
    Vr2vec = Vr2 * r2unit

    # Tangential component
    Vtan1 = sigma * gamma * (z + q * x) / r1
    Vtan1vec = Vtan1 * th1unit
    Vtan2 = sigma * gamma * (z + q * x) / r2
    Vtan2vec = Vtan2 * th2unit

    # Cartesian velocity
    V1 = Vtan1vec + Vr1vec
    V2 = Vtan2vec + Vr2vec

    exitflag = 1

    # Semi-major axis and extremal distances
    a = s / 2 / (1 - x ** 2)
    extremal_distances = minmax_distances(r1vec, r1, r1vec, r2, dth, a, V1, V2, m, muC)

    return V1, V2, extremal_distances, exitflag


def lancaster_blanchard(x, q, m):
    """Compute T(x) and its first three derivatives for the Lancaster-Blanchard formulation."""
    # Protection against invalid input
    if x < -1:
        x = abs(x) - 2
    elif x == -1:
        x = x + np.finfo(float).eps

    E = x * x - 1

    if x == 1:  # exactly parabolic
        T = 4.0 / 3 * (1 - q ** 3)
        Tp = 4.0 / 5 * (q ** 5 - 1)
        Tpp = Tp + 120.0 / 70 * (1 - q ** 7)
        Tppp = 3 * (Tpp - Tp) + 2400.0 / 1080 * (q ** 9 - 1)

    elif abs(x - 1) < 1e-2:  # near-parabolic, use series
        sig1, dsigdx1, d2sigdx21, d3sigdx31 = sigmax(-E)
        sig2, dsigdx2, d2sigdx22, d3sigdx32 = sigmax(-E * q * q)
        T = sig1 - q ** 3 * sig2
        Tp = 2 * x * (q ** 5 * dsigdx2 - dsigdx1)
        Tpp = Tp / x + 4 * x ** 2 * (d2sigdx21 - q ** 7 * d2sigdx22)
        Tppp = 3 * (Tpp - Tp / x) / x + 8 * x * x * (q ** 9 * d3sigdx32 - d3sigdx31)

    else:  # general case
        y = np.sqrt(abs(E))
        z = np.sqrt(1 + q ** 2 * E)
        f = y * (z - q * x)
        g = x * z - q * E

        if E < 0:
            d = np.arctan2(f, g) + np.pi * m
        elif E == 0:
            d = 0.0
        else:
            d = np.log(max(0, f + g))

        T = 2 * (x - q * z - d / y) / E
        Tp = (4 - 4 * q ** 3 * x / z - 3 * x * T) / E
        Tpp = (-4 * q ** 3 / z * (1 - q ** 2 * x ** 2 / z ** 2) - 3 * T - 3 * x * Tp) / E
        Tppp = (4 * q ** 3 / z ** 2 * ((1 - q ** 2 * x ** 2 / z ** 2) +
                2 * q ** 2 * x / z ** 2 * (z - x)) - 8 * Tp - 7 * x * Tpp) / E

    return T, Tp, Tpp, Tppp


def sigmax(y):
    """Series approximation to T(x) and its derivatives (near-parabolic cases)."""
    an = _AN

    # Powers of y: y^1, y^2, ..., y^25
    powers = np.array([y ** k for k in range(1, 26)])

    # sigma
    sig = 4.0 / 3 + np.dot(powers, an)

    # dsigma/dx
    coeffs1 = np.arange(1, 26).astype(float)
    prev_powers = np.concatenate(([1.0], powers[:-1]))
    dsigdx = np.dot(coeffs1 * prev_powers, an)

    # d2sigma/dx2
    coeffs2 = np.arange(0, 25).astype(float)
    if y != 0:
        prev_powers2 = np.concatenate(([1.0 / y, 1.0], powers[:-2]))
    else:
        prev_powers2 = np.concatenate(([0.0, 1.0], powers[:-2]))
    d2sigdx2 = np.dot(coeffs1 * coeffs2 * prev_powers2, an)

    # d3sigma/dx3
    coeffs3 = np.arange(-1, 24).astype(float)
    if y != 0:
        prev_powers3 = np.concatenate(([1.0 / (y * y), 1.0 / y, 1.0], powers[:-3]))
    else:
        prev_powers3 = np.concatenate(([0.0, 0.0, 1.0], powers[:-3]))
    d3sigdx3 = np.dot(coeffs1 * coeffs2 * coeffs3 * prev_powers3, an)

    return sig, dsigdx, d2sigdx2, d3sigdx3


def minmax_distances(r1vec, r1, r2vec, r2, dth, a, V1, V2, m, muC):
    """Compute minimum and maximum distances to the central body."""
    minimum_distance = min(r1, r2)
    maximum_distance = max(r1, r2)

    longway = abs(dth) > np.pi

    # Eccentricity vector (triple product identity)
    evec = (np.dot(V1, V1) * r1vec - np.dot(V1, r1vec) * V1) / muC - r1vec / r1

    e = np.linalg.norm(evec)

    # Apses
    pericenter = a * (1 - e)
    apocenter = np.inf
    if e < 1:
        apocenter = a * (1 + e)

    if m > 0:
        # Both apses always traversed for multi-revolution
        minimum_distance = pericenter
        maximum_distance = apocenter
    else:
        # Compute theta1 & theta2
        pm1 = np.sign(r1 * r1 * np.dot(evec, V1) - np.dot(r1vec, evec) * np.dot(r1vec, V1))
        pm2 = np.sign(r2 * r2 * np.dot(evec, V2) - np.dot(r2vec, evec) * np.dot(r2vec, V2))

        if e > 0:
            theta1 = pm1 * np.arccos(max(-1.0, min(1.0, np.dot(r1vec / r1, evec / e))))
            theta2 = pm2 * np.arccos(max(-1.0, min(1.0, np.dot(r2vec / r2, evec / e))))
        else:
            theta1 = 0.0
            theta2 = 0.0

        if theta1 * theta2 < 0:
            if abs(abs(theta1) + abs(theta2) - dth) < 5 * np.finfo(float).eps * abs(dth):
                minimum_distance = pericenter
            else:
                maximum_distance = apocenter
        elif longway:
            minimum_distance = pericenter
            if e < 1:
                maximum_distance = apocenter

    return np.array([minimum_distance, maximum_distance])
