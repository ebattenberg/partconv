

%module partconv
%{
#include "partconv.h"
%}

%include partconv.h

%include "carrays.i"
%array_class(int, intArray)
%array_class(double, doubleArray)


