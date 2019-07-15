/*
 * types.h
 *
 *  Created on: 15/07/2019
 *      Author: itallo
 */

#ifndef TYPES_H_
#define TYPES_H_

#pragma once

#include <tiffio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

// CONSTANTS DECLARATION

// Epsilon
const double EPS = 1e-7;

// Not a number
const double NaN = -sqrt(-1.0);

// Pi
const double PI = acos(-1);

// Karman's constant
const double VON_KARMAN = 0.41;

// Earth's gravity
const double GRAVITY = 9.81;

// Atmospheric density
const double RHO = 1.15;

// Specific heat of air
const double SPECIFIC_HEAT_AIR = 1004;

// Solar constant
const double GSC = 0.082;




#endif /* TYPES_H_ */
