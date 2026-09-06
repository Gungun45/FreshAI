import { NextRequest, NextResponse } from 'next/server';
import { Storage, getUserRoleForDocument } from '@/lib/storage';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId') || '';

    const doc = Storage.getById(id);
    if (!doc) {
      return NextResponse.json({ success: false, error: 'Document not found' }, { status: 404 });
    }

    const userRole = userId ? getUserRoleForDocument(doc, userId) : 'viewer';

    return NextResponse.json({
      success: true,
      document: doc,
      userRole: userRole || (doc.ownerId === userId ? 'owner' : null),
    });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message || 'Failed to fetch document' }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { title, content, userId } = body;

    if (!userId) {
      return NextResponse.json({ success: false, error: 'userId is required for validation' }, { status: 400 });
    }

    const updatedDoc = Storage.update(id, { title, content, userId });
    if (!updatedDoc) {
      return NextResponse.json({ success: false, error: 'Document not found' }, { status: 404 });
    }

    return NextResponse.json({ success: true, document: updatedDoc });
  } catch (error: any) {
    const isUnauthorized = error.message?.includes('Unauthorized');
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to update document' },
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
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json({ success: false, error: 'userId is required to delete' }, { status: 400 });
    }

    const deleted = Storage.delete(id, userId);
    return NextResponse.json({ success: deleted });
  } catch (error: any) {
    const isUnauthorized = error.message?.includes('Unauthorized');
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to delete document' },
      { status: isUnauthorized ? 403 : 500 }
    );
  }
}
