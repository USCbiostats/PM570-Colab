from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import jax
import jax.lax as lax
import jax.nn as nn
import jax.numpy as jnp
import jax.scipy as jsp

from jax.tree_util import register_pytree_node, register_pytree_node_class


def gradient_descent(loss, x0, step_size, tol=1e-3, maxiter=100):
    """TODO: add step size..."""
    
    loss_i = loss(x0)
    _lgrad = jax.grad(loss)

    def _cond_f(state):
        loss_imo, loss_i, iter_i, x_i = state
        diff_check = jnp.fabs(loss_i - loss_imo) > tol
        iter_check = iter_i < maxiter
        return jnp.all(jnp.array([diff_check, iter_check]))

    def _body_f(state):
        loss_imt, loss_imo, iter_imo, x_imo = state

        b = _lgrad(x_imo)

        d_i = -step_size * b
        x_i = x_imo + d_i
        loss_i = loss(x_i)

        return (loss_imo, loss_i, iter_imo + 1, x_i)

    state = (10000, loss_i, 0, x0)
    state = lax.while_loop(_cond_f, _body_f, state)

    return state


def newton_cg(loss, x0, step_size, tol=1e-3, maxiter=100):
    """TODO: add step size..."""

    loss_i = loss(x0)
    _lgrad = jax.grad(loss)

    def _cond_f(state):
        loss_imo, loss_i, iter_i, x_i = state
        diff_check = jnp.fabs(loss_i - loss_imo) > tol
        iter_check = iter_i < maxiter
        return jnp.all(jnp.array([diff_check, iter_check]))

    def _body_f(state):
        loss_imt, loss_imo, iter_imo, x_imo = state

        b = _lgrad(x_imo)

        # lets define matrix vector product over the hessian
        # but without having to explicitly compute hessian...
        # we want to compute inv(H(x_imo)) @ grad(x_imo)
        def _matvec(d):
            return _hvp(_lgrad, x_imo, d)

        d_i, _ = jsp.sparse.linalg.cg(_matvec, -b)
        x_i = x_imo + d_i
        loss_i = loss(x_i)

        return (loss_imo, loss_i, iter_imo + 1, x_i)

    state = (10000, loss_i, 0, x0)
    state = lax.while_loop(_cond_f, _body_f, state)

    return state


@register_pytree_node_class
class AbstractRegressionFunc(ABC):
    """Abstract class that lets us pass around a function
    into JIT compatible calls
    """

    @abstractmethod
    def __call__(self, data, x) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    def tree_flatten(self):
        children = ()
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


def _pad_variant(x: jnp.ndarray, covar: jnp.ndarray) -> jnp.ndarray:
    """Pad the genotype to include covariates"""
    X = jnp.hstack((x[:, jnp.newaxis], covar))
    return X


# OLS using QR decomp with variables (x, covar) and y
class OLSQR(AbstractRegressionFunc):
    def __call__(self, data, x):
        y, covar = data
        X = _pad_variant(x, covar)

        q_matrix, r_matrix = jnp.linalg.qr(X, mode="reduced")
        qty = q_matrix.T @ y
        beta_hat = jsp.linalg.solve_triangular(r_matrix, qty)
        df = q_matrix.shape[0] - q_matrix.shape[1]
        residual = y - q_matrix @ qty
        rss = jnp.sum(residual**2)
        sigma = jnp.sqrt(rss / df)
        se = (
            jnp.sqrt(
                jnp.diag(
                    jsp.linalg.cho_solve((r_matrix, False), jnp.eye(r_matrix.shape[0]))
                )
            )
            * sigma
        )
        t_scores = beta_hat / se
        # p_value = jnp.array(2 * stats.t.sf(abs(t_scores), df=df))
        log_p_value = jnp.log(2) + jsp.stats.norm.logcdf(-jnp.abs(t_scores))

        return data, jnp.array([beta_hat, se, t_scores, log_p_value]).T[0]


# OLS using jax built in least-squares function
class OLSJAX(AbstractRegressionFunc):
    def __call__(self, data, x):
        y, covar = data
        n = len(y)
        X = _pad_variant(x, covar)

        beta_hat, rss, rank, svals = jnp.linalg.lstsq(X, y)

        se = jnp.sqrt(rss / (n - rank)) / svals
        z_scores = beta_hat / se
        log_p_value = jnp.log(2) + jsp.stats.norm.logcdf(-jnp.abs(z_scores))

        return data, jnp.array([beta_hat, se, z_scores, log_p_value]).T[0]


def _hvp(g, primals, tangents):
    # hessian vector product
    return jax.jvp(g, (primals,), (tangents,))[1]



def _bern_negloglike(beta, y, X):
    # squash R => [0, 1]
    lin_pred = X @ beta

    # prob = nn.sigmoid(lin_pred)
    # lets break this down...
    # loglikelihood is sum_i y_i * log(prob_i) + (1 - y_i) * log(1 - prob_i)
    # log(prob_i) = log(sigmoid(lin_pred)) = log(1) - log(1 + e^(-lin_pred))
    #             = 0 - log(1 + e^(-lin_pred)) = -softplus(-lin_pred)
    # similarly, 1 - prob_i = 1 - sigmoid(lin_pred) = sigmoid(-lin_pred)
    # hence, log(1 - prob_i) = log(sigmoid(-lin_pred)) = 0 - log(1 + e^lin_pred)
    #                        = -softplus(lin_pred)
    # we drop the "-" in front of softplus to reflect neg log like
    return jnp.sum(nn.softplus(jnp.where(y, -lin_pred, lin_pred)))


class LogisticRegression(AbstractRegressionFunc):
    def __init__(self, tol=1e-3, maxiter=100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(self, data, x):

        y, covar = data
        X = _pad_variant(x, covar)

        _loss = partial(_bern_negloglike, y=y, X=X)
        optstate = _newton_cg(
            _loss, jnp.zeros(X.shape[1]), tol=self.tol, maxiter=self.maxiter
        )

        loss_imo, loss_i, iter_i, x_i = optstate
        beta_hat = x_i
        converged = jnp.all(
            jnp.array(
                [(jnp.fabs(loss_i - loss_imo) < self.tol), (iter_i < self.maxiter)]
            )
        )
        converged = (jnp.ones(X.shape[1]) * converged).astype(float)

        # not exactly the fisher information matrix, but close enough...
        se = jnp.sqrt(jnp.diag(jnp.linalg.inv(jax.hessian(_loss)(beta_hat))))
        t_scores = beta_hat / se
        # p_value = jnp.array(2 * stats.t.sf(abs(t_scores), df=df))
        log_p_value = jnp.log(2) + jsp.stats.norm.logcdf(-jnp.abs(t_scores))

        return data, jnp.array([beta_hat, se, t_scores, log_p_value, converged]).T[0]

    def tree_flatten(self):
        children = (self.tol, self.maxiter)
        aux = ()
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
