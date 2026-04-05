import torch

def find_field_axis(shape, n_fields):

    axlist = [i for i, s in enumerate(shape) if s == n_fields]

    assert len(axlist) == 1, f"Ambiguous field axis for automatic tensor cast (matches {axlist}); should only being one matching axis"

    return axlist[0]



def my_cast(t,x):
    '''
    Casts a 1-D tensor t into the same shape as x.
    '''

    assert t.ndim==1, "Automatic Tensor Cast only supported for tensor dimension 1"

    ax = find_field_axis( x.shape, t.numel() )

    view_shape = [1] * x.ndim
    view_shape[ax] = t.shape[0]

    t_cast = t.view(view_shape).expand_as(x)

    return t_cast

def slice_by_field(x, start, stop, n_fields):

    ax = find_field_axis( x.shape, n_fields )

    slices = [slice(None)] * x.ndim
    slices[ax] = slice(start, stop)

    return tuple(slices)

def pick_field(x, idx, n_fields):

    ax = find_field_axis( x.shape, n_fields )

    slices = [slice(None)] * x.ndim
    slices[ax] = idx

    return tuple(slices)



def check_axis_shape_conflict( checked_params, forbidden_params ):

    assert isinstance(checked_params, dict), "checked_params must be dict"
    assert isinstance(forbidden_params, dict), "forbidden_params must be dict"

    for k_chk, v_chk in checked_params.items():
        for k_vbt, v_vbt in forbidden_params.items():
            
            if v_chk == v_vbt:
                raise ValueError(f"To ensure unambiguous automatic tensor cast, {k_chk} and {k_vbt} must be different (both {v_chk})")

    for k1, v1 in forbidden_params.items():
        for k2, v2 in forbidden_params.items():

            if k1 != k2 and v1 == v2:
                raise ValueError(f"To ensure unambiguous automatic tensor cast, {k1} and {k2} must be different (both {v1})")

