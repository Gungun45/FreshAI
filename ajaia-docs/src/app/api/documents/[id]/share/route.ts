import { NextRequest, NextResponse } from 'next/server';
import { Storage } from '@/lib/storage';

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { requestingUserId, targetEmail, targetRole } = body;

    if (!requestingUserId || !targetEmail || !targetRole) {
      return NextResponse.json(
        { success: false, error: 'requestingUserId, targetEmail, and targetRole are required' },
        { status: 400 }
      );
    }

    if (targetRole !== 'editor' && targetRole !== 'viewer') {
      return NextResponse.json(
        { success: false, error: 'Role must be either "editor" or "viewer"' },
        { status: 400 }
      );
    }

    const updatedDoc = Storage.share(id, requestingUserId, targetEmail, targetRole);
    return NextResponse.json({ success: true, document: updatedDoc });
  } catch (error: any) {
    const isUnauthorized = error.message?.includes('Unauthorized');
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to share document' },
      { status: isUnauthorized ? 403 : 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const requestingUserId = searchParams.get('requestingUserId');
    const targetUserId = searchParams.get('targetUserId');

    if (!requestingUserId || !targetUserId) {
      return NextResponse.json(
        { success: false, error: 'requestingUserId and targetUserId are required' },
        { status: 400 }
      );
    }

    const updatedDoc = Storage.removeCollaborator(id, requestingUserId, targetUserId);
    return NextResponse.json({ success: true, document: updatedDoc });
  } catch (error: any) {
    const isUnauthorized = error.message?.includes('Unauthorized');
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to remove collaborator' },
      { status: isUnauthorized ? 403 : 500 }
    );
  }
}
