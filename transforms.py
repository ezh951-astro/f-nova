import torch

from util.transform_helper import *

class Preprocessor():
    """Class containing functions related to transforming the data into coordinates consumable by models."""

    def __init__(self,gp,device):

        self.gp = gp
        self.device = device

        self.mu = torch.load(gp.paths.mu, map_location=device)
        self.sigma = torch.load(gp.paths.sigma, map_location=device)

    def preprocess(self,x):
        """Applies normalization and log/asinh scaling to raw data tensor with automatic axis detection."""

        gp = self.gp
        x = x.to(self.device)

        nF = gp.data_summary.n_fields

        if gp.preprocess.log:
            x[slice_by_field(x,0,3,nF)] = torch.log10(x[slice_by_field(x,0,3,nF)])
        if gp.preprocess.asinh:
            x[slice_by_field(x,3,6,nF)] = torch.asinh(x[slice_by_field(x,3,6,nF)])

        x = self.normalize(x)

        return x

    def postprocess(self,x):
        """Applies un-normalization and log/asinh un-scaling (i.e., exp/sinh) to raw data tensor with automatic axis detection."""

        gp = self.gp
        x = x.to(self.device)

        nF = gp.data_summary.n_fields

        x = self.unnormalize(x)

        if gp.preprocess.asinh:
            x[slice_by_field(x,3,6,nF)] = torch.sinh(x[slice_by_field(x,3,6,nF)])
        if gp.preprocess.log:
            x[slice_by_field(x,0,3,nF)] = 10.**(x[slice_by_field(x,0,3,nF)])

        return x

    def normalize(self,x):
        """Applies normalization to raw data tensor with automatic axis detection."""

        x = x.to(self.device)

        mu_cast = my_cast(self.mu,x).to(self.device)
        sigma_cast = my_cast(self.sigma,x).to(self.device)

        return (x - mu_cast) / sigma_cast

    def unnormalize(self,x):
        """Applies un-normalization to raw data tensor with automatic axis detection."""

        x = x.to(self.device)

        mu_cast = my_cast(self.mu,x).to(self.device)
        sigma_cast = my_cast(self.sigma,x).to(self.device)

        return sigma_cast * x + mu_cast



class Physics():
    """Class containing functions related to physical quantities."""

    def __init__(self,gp,device):

        self.gp = gp
        self.device = device

        self.nF = gp.data_summary.n_fields
        self.boxsize = gp.data_summary.box_size
        self.boxdims = gp.data_summary.box_dims

        self.idg = torch.load(gp.paths.ideal_constant, map_location=device)

        self.__dict__.update(gp.physics.__dict__)

    def sum_mass(self, x, pretransformed=True): # pretransformed means in log/asinh coords
        """Sums the mass field over spatial dimensions.

        Extracts mass from ``x``, undoes log preprocessing if needed,
        and sums over dimensions matching ``box_size``.

        Args:
            x: Input tensor containing all fields.
            pretransformed: Whether ``x`` is in log/asinh coordinates.
        """

        x = x.to(self.device)

        mass = x[ pick_field(x, 0, self.nF) ]

        if pretransformed and self.gp.preprocess.log:
            mass = 10.**mass

        dims = tuple(i for i, s in enumerate(mass.shape) if s==self.gp.data_summary.box_size )
        assert len(dims)==self.boxdims, f"Should be {self.boxdims} boxsize dims"

        return mass.sum(dim=dims)

    def sum_momentum(self, x, pretransformed=True):
        """Sums the momentum field over spatial dimensions.

        Extracts mass from ``x``, undoes log/asinh preprocessing if needed,
        and sums over dimensions matching ``box_size``.

        Args:
            x: Input tensor containing all fields.
            pretransformed: Whether ``x`` is in log/asinh coordinates.
        """

        x = x.to(self.device)

        mass_slice     = slice_by_field(x,0,1,self.nF)
        velocity_slice = slice_by_field(x,3,6,self.nF)

        if pretransformed and self.gp.preprocess.log:
            mass = 10. ** x[mass_slice]
        else:
            mass = x[mass_slice]

        if pretransformed and self.gp.preprocess.asinh:
            velocity = torch.sinh( x[velocity_slice] )
        else:
            velocity = x[velocity_slice]

        mass = mass.expand_as(velocity)

        momentum = mass * velocity

        dims = tuple(i for i, s in enumerate(momentum.shape) if s==self.gp.data_summary.box_size )
        assert len(dims)==self.boxdims, f"Should be {self.boxdims} boxsize dims"

        return momentum.sum(dim=dims)

    def ideal_gas(self, x, pretransformed=True):
        """Computes the ideal gas law residual. 
        
        For a perfect ideal gas returns ones, or in log/asinh coordinates, zeros.

        Args:
            x: Input tensor containing all fields.
            pretransformed: Whether ``x`` is in log/asinh coordinates.
        """

        x = x.to(self.device)

        idg = self.idg.to(x.device)

        if pretransformed and self.gp.preprocess.log:
            return x[pick_field(x,1,self.nF)] - x[pick_field(x,0,self.nF)] - x[pick_field(x,2,self.nF)] - idg
        else:
            assert (x != 0).all(), "No zeros allowed in density, pressure, temperature"
            return x[pick_field(x,1,self.nF)] / x[pick_field(x,0,self.nF)] / x[pick_field(x,2,self.nF)] / idg
