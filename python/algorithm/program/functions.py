def exp2model(e):
    # C[i] * X^n * Y^m
    return ' + '.join([
        f'C[{i}]' +
        ('*' if x>0 or y>0 else '') +
        (f'X^{x}' if x>1 else 'X' if x==1 else '') +
        ('*' if x>0 and y>0 else '') +
        (f'Y^{y}' if y>1 else 'Y' if y==1 else '')
        for i,(x,y) in enumerate(e)
    ])




