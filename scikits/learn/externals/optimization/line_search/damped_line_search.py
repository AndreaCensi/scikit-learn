
"""
A damped line search
"""

class DampedLineSearch(object):
  """
  A damped line search, takes a point and a direction.
  Tests a new point for minimization, if it is greater that the current cost times (1 + error),
    the step is divided by two, until the step is too small
  """
  def __init__(self, min_alpha_step, damped_error, alpha_step = 1., **kwargs):
    """
    Needs to have :
      - a minimum step size (min_alpha_step)
      - a factor to allow the cost to rise a little bit (damped_error)
    Can have :
      - a step modifier, a factor to modulate the step (alpha_step = 1.)
    """
    self.min_step_size = min_alpha_step
    self.damped_error = damped_error
    self.step_size = alpha_step

  def __call__(self, origin, function, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - function is the function to minimize
      - state is the state of the optimizer
    """
    direction = state['direction']
    if 'initial_alpha_step' in state:
      step_size = state['initial_alpha_step']
    else:
      step_size = self.step_size
    currentValue = function(origin)
    optimalPoint = origin + step_size * direction
    newValue = function(optimalPoint)

    while(newValue > currentValue * (1. + self.damped_error)):
      step_size /= 2.
      if step_size < self.min_step_size:
        break
      optimalPoint = origin + step_size * direction
      newValue = function(optimalPoint)
    else:
      state['alpha_step'] = step_size
      return optimalPoint
    state['alpha_step'] = 0.
    return origin

