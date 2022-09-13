import numpy as np

""" 
A set of functions for calculating joint probabilities of dependant binary outcomes given the marginal probabilities.
These are technically Guassian Copulas (equation 2 of Lin & Chaganty)
This is based of of the works of:
https://jsdajournal.springeropen.com/articles/10.1186/s40488-021-00118-z
https://arxiv.org/ftp/arxiv/papers/2101/2101.02280.pdf
"""


def copula_clip(u1, u2, vec):
    """
    A function used to ensure that copulas are restricted to outputting values in the correct bounds
    For details see
    http://www.nematrian.com/CountermonotonicityCopula
    http://www.nematrian.com/ComonotonicityCopula
    :param u1: probability of event 1 not happening
    :param u2: probability of event 2 not happening
    :param vec: copula suggested probability of neither event 1 nor event 2 happening
    :return: correctly bounded probability of neither event 1 nor event 2 happening
    """
    upper = np.min((u1, u2), axis=0)
    lower = np.clip(u1 + u2 - 1, 0, None)
    return np.clip(vec, lower, upper)


def cqq(q1, q2, r):
    """
    Backbone of the copula calculations, for details see:
    https://jsdajournal.springeropen.com/articles/10.1186/s40488-021-00118-z
    https://arxiv.org/ftp/arxiv/papers/2101/2101.02280.pdf
    :param q1: probability of event 1 not happening
    :param q2: probability of event 2 not happening
    :param r: correlation between events 1 and 2
    :return: probability of neither event 1 nor event 2 happening
    """
    np.clip(r, -1, 1)
    p1, p2 = 1 - q1, 1 - q2
    return_vec = 1 - p1 - p2 + p1 * p2 + r * np.sqrt(p1 * p2 * q1 * q2)
    return_vec = copula_clip(q1, q2, return_vec)
    return return_vec


def bivariate_00(p1, p2, r):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param r: Correlation of event 1 and 2
    :return: Joint probability of neither event occurring
    """
    q1, q2 = 1 - p1, 1 - p2
    return_vec = cqq(q1, q2, r)
    return return_vec


def bivariate_10(p1, p2, r):
    """
     :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param r: Correlation of event 1 and 2
    :return: Joint probability of event 1 and not event 2 occurring
    """
    q1, q2 = 1 - p1, 1 - p2
    return_vec = q2 - cqq(q1, q2, r)
    return return_vec


def bivariate_01(p1, p2, r):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param r: Correlation of event 1 and 2
    :return: Joint probability of event 2 and not event 1 occurring
    """
    q1, q2 = 1 - p1, 1 - p2
    return_vec = q1 - cqq(q1, q2, r)
    return return_vec


def bivariate_11(p1, p2, r):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param r: Correlation of event 1 and 2
    :return: Joint probability both events occurring
    """
    q1, q2 = 1 - p1, 1 - p2
    return_vec = 1 - q1 - q2 + cqq(q1, q2, r)
    return return_vec


def bivariate_any(p1, p2, r):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param r: Correlation of event 1 and 2
    :return: Probability of either event occuring
    """
    return 1 - bivariate_00(p1, p2, r)


def bivariate_all(p1, p2, r):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param r: Correlation of event 1 and 2
    :return: Joint probability both events occurring
    """
    return bivariate_11(p1, p2, r)


def trivariate000(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability none of the three events occurring
    """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q10 = C12 / q2
    q30 = C23 / q2
    C130 = cqq(q10, q30, r3)

    return_vec = q2 * C130
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate001(p1, p2, p3, r1, r2, r3, r4):
    """
        :param p1: Marginal probability of event 1
        :param p2: Marginal probability of event 2
        :param p3: Marginal probability of event 3
        :param r1: Correlation of event 1 and 2
        :param r2: Correlation of event 2 and 3
        :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
        :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
        :return: Joint probability of only event 3 occurring
        """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q10 = C12 / q2
    q30 = C23 / q2
    C130 = cqq(q10, q30, r3)

    return_vec = C12 - q2 * C130
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate010(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of only event 2 occurring
    """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q11 = (q1 - C12) / p2
    q31 = (q3 - C23) / p2

    C131 = cqq(q11, q31, r4)

    return_vec = p2 * C131
    return_vec = np.where(np.isclose(p2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate011(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of only events 2 and 3 occurring
        """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q11 = (q1 - C12) / p2
    q31 = (q3 - C23) / p2
    C131 = cqq(q11, q31, r4)

    return_vec = q1 - C12 - p2 * C131
    return_vec = np.where(np.isclose(p2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate100(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of only event 1 occurring
            """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q10 = C12 / q2
    q30 = C23 / q2
    C130 = cqq(q10, q30, r3)

    return_vec = C23 - q2 * C130
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate101(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of only events 1 and 3 occurring
    """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q10 = C12 / q2
    q30 = C23 / q2
    C130 = cqq(q10, q30, r3)

    return_vec = q2 - C23 - C12 + q2 * C130
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate110(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of only events 1 and 2 occurring
            """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q11 = (q1 - C12) / p2
    q31 = (q3 - C23) / p2
    C131 = cqq(q11, q31, r4)

    return_vec = q3 - C23 - p2 * C131
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate111(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of all events occurring
            """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q11 = (q1 - C12) / p2
    q31 = (q3 - C23) / p2
    C131 = cqq(q11, q31, r4)

    return_vec = 1 - q1 - q2 - q3 + C12 + C23 + p2 * C131
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec


def trivariate_any(p1, p2, p3, r1, r2, r3, r4):
    """
    :param p1: Marginal probability of event 1
    :param p2: Marginal probability of event 2
    :param p3: Marginal probability of event 3
    :param r1: Correlation of event 1 and 2
    :param r2: Correlation of event 2 and 3
    :param r3: Correlation of event 1 and 3 conditional on event 2 not occurring
    :param r4: Correlation of event 1 and 3 conditional on event 2 occurring
    :return: Joint probability of at least one event occurring
            """
    q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

    C12 = cqq(q1, q2, r1)
    C23 = cqq(q2, q3, r2)

    q10 = C12 / q2
    q30 = C23 / q2
    C130 = cqq(q10, q30, r3)

    return_vec = 1 - q2 * C130
    return_vec = np.where(np.isclose(q2, 0.0), 0.0, return_vec)
    return return_vec
