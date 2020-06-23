[
    OrderedDict([
        ('0', PReLU(num_parameters=32)),
        ('1', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=32)),
        ('3', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=48)),
        ('1', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=48)),
        ('3', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=32)),
        ('1', Conv2d(32, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
        ('2', PReLU(num_parameters=48)),
        ('3', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=64)),
        ('1', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=64)),
        ('3', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=48)),
        ('1', Conv2d(48, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
        ('2', PReLU(num_parameters=64)),
        ('3', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=64)),
        ('1', Conv2d(64, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=512)),
        ('2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=512)),
        ('4', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=512)),
        ('2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=512)),
        ('4', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=64)),
        ('1', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=64)),
        ('3', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=512)),
        ('2', Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=64)),
        ('4', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=48)),
        ('1', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=48)),
        ('3', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=64)),
        ('2', Conv2d(64, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=48)),
        ('4', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=32)),
        ('1', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=32)),
        ('3', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=48)),
        ('2', Conv2d(48, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=32)),
        ('4', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=512)),
        ('2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=512)),
        ('4', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=512)),
        ('1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=512)),
        ('3', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=512)),
        ('2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=512)),
        ('4', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=64)),
        ('1', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=64)),
        ('3', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=512)),
        ('2', Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=64)),
        ('4', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=48)),
        ('1', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=48)),
        ('3', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=64)),
        ('2', Conv2d(64, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=48)),
        ('4', Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', PReLU(num_parameters=32)),
        ('1', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('2', PReLU(num_parameters=32)),
        ('3', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]),
    OrderedDict([
        ('0', Upsample(scale_factor=2.0, mode=bilinear)),
        ('1', PReLU(num_parameters=48)),
        ('2', Conv2d(48, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('3', PReLU(num_parameters=32)),
        ('4', Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))])]
