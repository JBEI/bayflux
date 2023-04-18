subroutine executeemuoperationsfortran(a, b, operations, fluxes)
    ! This subroutine injects fluxes into matrix a and b according to the
    ! coordinates and multipliers specified in 'operations'
    ! Note that a and b are modified in place, and should be pre-initialized
    ! with all zeros.

    ! declare variable types
    real(kind=8), dimension(:,:),intent(in) :: operations
    real(kind=8), dimension(:),intent(in) :: fluxes
    real(kind=8), dimension(:,:),intent(inout) :: a, b
    real(kind=8) :: value
    integer :: o, i, j
    
    ! loop over rows in operations
    do o = lbound(operations,1), ubound(operations,1)
    
        ! compute new value to inject
        value = fluxes(int(operations(o,4)) + 1) * operations(o,5)

        ! get matrix coordinates of where to add/inject value
        i = int(operations(o,2)) + 1
        j = int(operations(o,3)) + 1
        
        if (int(operations(o,1)) == 0) then ! inject in A matrix
            a(i,j) = a(i,j) + value
        else ! inject in B matrix
            b(i,j) = b(i,j) + value
        end if
    end do
    
end subroutine
