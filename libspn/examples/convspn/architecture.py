from libspn.graph.localsum import LocalSum
from libspn.graph.convprod2d import ConvProd2D
from libspn.graph.convproddepthwise import ConvProdDepthWise
from libspn.graph.parsums import ParSums
from libspn.graph.sum import Sum


def mnist_wicker(in_var, num_channels_prod, num_channels_sums, num_classes=10):
    prod0 = ConvProd2D(
        in_var, num_channels=num_channels_prod[0], padding='valid', kernel_size=2, strides=2,
        grid_dim_sizes=[28, 28])
    sum0 = LocalSum(prod0, num_channels=num_channels_sums[0])
    prod1 = ConvProdDepthWise(sum0, padding='valid', kernel_size=2, strides=2)
    sum1 = LocalSum(prod1, num_channels=num_channels_sums[1])
    prod2 = ConvProdDepthWise(sum1, padding='full', kernel_size=2, strides=1)
    sum2 = LocalSum(prod2, num_channels=num_channels_sums[2])
    prod3 = ConvProdDepthWise(sum2, padding='full', kernel_size=2, strides=1, dilation_rate=2)
    sum3 = LocalSum(prod3, num_channels=num_channels_sums[3])
    prod4 = ConvProdDepthWise(sum3, padding='full', kernel_size=2, strides=1, dilation_rate=4)
    sum4 = LocalSum(prod4, num_channels=num_channels_sums[4])
    prod5 = ConvProdDepthWise(sum4, padding='final', kernel_size=2, strides=1, dilation_rate=8)
    class_roots = ParSums(prod5, num_sums=num_classes)
    root = Sum(class_roots)
    return root, class_roots
